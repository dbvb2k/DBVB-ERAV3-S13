import os
import math
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from torch.utils.data import Dataset, DataLoader
import tiktoken
from model135m import SmolLM2ForCausalLM135M, SmolLM2Config135M
from transformers import AutoTokenizer
from typing import Optional
from datetime import datetime
import logging
from tqdm.auto import tqdm
import time
from datetime import timedelta
import torch.backends.cuda as cuda
import torch.cuda
from torch.nn.functional import scaled_dot_product_attention
from datasets import load_dataset
from itertools import islice
import signal
import sys
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
import yaml
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

NUM_DASHES = 100

def setup_training_environment():
    """Setup training environment including CUDA and precision settings"""
    # Set high precision matrix multiplication (only once)
    torch.set_float32_matmul_precision('high')
    logging.info("Set float32 matmul precision to high")
    
    # Set up CUDA settings
    if torch.cuda.is_available():
        logging.info("CUDA is available")
        logging.info(f"Using device: {torch.cuda.get_device_name()}")
        logging.info(f"CUDA version: {torch.version.cuda}")
        
        # Enable CUDA synchronization for accurate timing
        torch.cuda.synchronize()
        logging.info("CUDA synchronization enabled")
    else:
        logging.warning("CUDA is not available, training on CPU")

class TextDataset(Dataset):
    def __init__(self, block_size: int = 128, vocab_size: int = 49152):
        logging.info(f"Initializing TextDataset with block_size={block_size}, vocab_size={vocab_size}")
        self.block_size = block_size
        self.vocab_size = vocab_size
        
        # Initialize tokenizer
        logging.info("-" * NUM_DASHES)
        logging.info("Initializing tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/cosmo2-tokenizer")
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        logging.info("Tokenizer initialized")
        
        # Load dataset with streaming
        logging.info("Loading dataset using HuggingFace datasets (streaming mode) ...")
        self.dataset = load_dataset(
            "HuggingFaceTB/smollm-corpus",
            "cosmopedia-v2",
            split="train",  # Explicitly specify split
            streaming=True
        )
        logging.info("Dataset loaded successfully")
        
        # Convert streaming dataset to list for length calculation and random access
        logging.info("Processing dataset...")
        self.processed_data = []
        total_sequences = 0
        total_examples = 0
        
        initial_examples = 1000
        # Create progress bar for initial examples
        pbar = tqdm(islice(self.dataset, initial_examples), desc="Processing examples", unit=" examples")
        
        try:
            for item in pbar:
                total_examples += 1
                # Tokenize with padding and truncation
                encoding = self.tokenizer(
                    item["text"],
                    max_length=self.block_size * 2,  # Allow longer sequences initially
                    truncation=True,
                    padding=False,
                    return_tensors=None  # Return list of ints
                )
                
                tokens = encoding
                if isinstance(tokens, dict):
                    tokens = tokens["input_ids"]
                
                # Process tokens in strided sequences
                stride = self.block_size // 2  # 50% overlap between sequences
                for i in range(0, len(tokens) - self.block_size, stride):
                    chunk = tokens[i:i + self.block_size + 1]
                    if len(chunk) == self.block_size + 1:
                        # Convert to tensor and validate
                        chunk_tensor = torch.tensor(chunk, dtype=torch.long)
                        if (chunk_tensor >= 0).all() and (chunk_tensor < self.vocab_size).all():
                            self.processed_data.append(chunk_tensor)
                            total_sequences += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'total_sequences': total_sequences,
                    'avg_seq_per_example': total_sequences/total_examples if total_examples > 0 else 0
                })
                
                # Break if we have enough sequences
                if total_sequences >= 1000:
                    break
                    
        except Exception as e:
            logging.error(f"Error processing dataset: {str(e)}")
            raise
            
        if not self.processed_data:
            raise ValueError("No sequences were processed. Check dataset and processing logic.")
            
        self.n_samples = len(self.processed_data)
        logging.info(f"Dataset processing complete. Total sequences: {self.n_samples}")
        logging.info(f"Average sequence length: {sum(len(seq) for seq in self.processed_data)/self.n_samples:.1f}")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        sequence = self.processed_data[idx]
        x = sequence[:-1]  # Input sequence
        y = sequence[1:]   # Target sequence
        return x, y

class SmolLM2LightningModule(pl.LightningModule):
    def __init__(
        self, 
        config: Optional[SmolLM2Config135M] = None,
        learning_rate: float = 0.003,  # From config
        total_steps: int = 2000000,    # From config
        warmup_steps: int = 2000,      # From config
        inference_interval: int = 500,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.learning_rate = learning_rate
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.inference_interval = inference_interval
        self.last_inference_step = -1
        
        # Initialize model
        self.model = SmolLM2ForCausalLM135M(config)
        self.tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/cosmo2-tokenizer")
        self.test_prompt = "Once upon a time"

    def generate_sample_text(self):
        """Generate sample text using the current model state"""
        logging.info("\nGenerating sample text...")
        logging.info(f"Prompt: {self.test_prompt}")
        
        try:
            # Tokenize the prompt with explicit attention mask
            inputs = self.tokenizer(
                self.test_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_position_embeddings,
                return_attention_mask=True
            )
            
            # Move inputs to device safely
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)
            
            # Set generation parameters
            gen_kwargs = {
                "max_length": 100,
                "num_return_sequences": 1,
                "temperature": 0.7,
                "top_p": 0.9,
                "do_sample": True,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "attention_mask": attention_mask,
                "use_cache": True,
                "no_repeat_ngram_size": 3,
                "early_stopping": True
            }
            
            # Generate text with error handling
            with torch.no_grad():
                try:
                    output_sequences = self.model.generate(
                        input_ids=input_ids,
                        **gen_kwargs
                    )
                    
                    # Decode and log the generated text
                    generated_text = self.tokenizer.decode(output_sequences[0], skip_special_tokens=True)
                    logging.info("Generated text:")
                    logging.info("-" * NUM_DASHES)
                    logging.info(generated_text)
                    logging.info("-" * NUM_DASHES)
                    
                    return generated_text
                    
                except RuntimeError as e:
                    logging.error(f"Error during generation: {str(e)}")
                    return None
                    
        except Exception as e:
            logging.error(f"Error during text generation setup: {str(e)}")
            return None

    def training_step(self, batch, batch_idx):
        try:
            if self.training_start_time is None:
                self.training_start_time = time.time()
                self.last_log_time = self.training_start_time
            
            step_start = time.time()
            x, y = batch
            
            # Forward pass
            outputs = self.model(input_ids=x, labels=y)
            loss = outputs.loss
            
            # Check for NaN loss
            if torch.isnan(loss) or torch.isinf(loss):
                logging.warning(f"NaN/Inf loss detected at step {self.trainer.global_step}. Skipping batch.")
                return None
            
            # Log metrics
            self.log('train_loss', loss.item(), prog_bar=True, on_step=True)
            current_lr = self.optimizers().param_groups[0]['lr']
            self.log('learning_rate', current_lr, prog_bar=True, on_step=True)
            
            # Calculate and log step time
            step_time = time.time() - step_start
            self.log('step_time', step_time, prog_bar=True, on_step=True)
            
            if self.trainer.global_step % 100 == 0:
                logging.info(f"Step {self.trainer.global_step}: loss = {loss.item():.4f}, lr = {current_lr:.10f}")
            
            # Only do inference on actual steps, not accumulated gradients
            global_step = self.trainer.global_step
            if (global_step % self.inference_interval == 0 and 
                global_step != self.last_inference_step and 
                global_step > 0):
                self.last_inference_step = global_step
                logging.info(f"\nRunning inference at step {global_step}:")
                self.generate_sample_text()
            
            return loss
            
        except Exception as e:
            logging.error(f"Error in training step: {str(e)}")
            return None

    def on_train_start(self):
        """Generate initial sample before training starts"""
        logging.info("\nGenerating initial sample before training:")
        self.generate_sample_text()

    def on_train_end(self):
        """Generate final sample after training completes"""
        logging.info("\nGenerating final sample after training:")
        self.generate_sample_text()

    def configure_optimizers(self):
        # Load config
        config = load_config()
        
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            betas=(
                config['optimizer']['optimizer_factory']['adam_beta1'],
                config['optimizer']['optimizer_factory']['adam_beta2']
            ),
            eps=config['optimizer']['optimizer_factory']['adam_eps'],
            weight_decay=config['optimizer']['weight_decay']
        )
        
        # Use linear schedule as specified in config
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.total_steps
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }

def signal_handler(signum, frame):
    print("\nReceived interrupt signal. Cleaning up...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    sys.exit(0)

def load_config(config_path: str = "config_smollm2_135M.yaml") -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_precision_from_config(dtype: str) -> str:
    """Convert config dtype to PyTorch Lightning precision format"""
    precision_mapping = {
        'bfloat16': 'bf16-mixed',
        'float16': '16-mixed',
        'float32': '32-true',
        'float64': '64-true'
    }
    return precision_mapping.get(dtype, '32-true')  # Default to 32-true if not found

def train_model(resume_from_checkpoint: Optional[str] = None, additional_steps: Optional[int] = None):
    # Load configuration
    config = load_config()
    
    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    start_time = time.time()
    logging.info("=" * NUM_DASHES)
    logging.info("Starting training process...")
    logging.info("=" * NUM_DASHES)
    
    # Setup training environment
    setup_training_environment()
    
    # Initialize model config with YAML settings
    model_config = SmolLM2Config135M(**config['model']['model_config'])
    
    # Initialize dataset with config settings
    dataset = TextDataset(
        block_size=config['tokens']['sequence_length'],
        vocab_size=config['model']['model_config']['vocab_size']
    )
    
    # Configure data loader based on config
    train_loader = DataLoader(
        dataset,
        batch_size=config['tokens']['micro_batch_size'],
        shuffle=True,
        num_workers=config['data_stages'][0]['data']['num_loading_workers'],
        pin_memory=False,
        drop_last=True
    )
    
    # Initialize model with config
    model = SmolLM2LightningModule(
        config=model_config,
        learning_rate=config['optimizer']['learning_rate_scheduler']['learning_rate'],
        total_steps=config['tokens']['train_steps'],
        warmup_steps=config['optimizer']['learning_rate_scheduler']['lr_warmup_steps'],
    )
    
    # Configure checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath='lightning_checkpoints',
        filename='smollm2-step{step}-loss{train_loss:.2f}',
        every_n_train_steps=config['checkpoints']['checkpoint_interval'],
        save_top_k=5,
        monitor='train_loss',
        mode='min',
        save_last=True
    )
    
    # Configure progress bar
    progress_bar = TQDMProgressBar(
        refresh_rate=1
    )
    
    # Configure trainer with config settings
    trainer = pl.Trainer(
        max_steps=config['tokens']['train_steps'],
        callbacks=[checkpoint_callback, progress_bar],
        accelerator='cpu',
        devices=1,
        precision=get_precision_from_config(config['model']['dtype']),  # Convert dtype to supported format
        enable_progress_bar=True,
        logger=True,
        log_every_n_steps=config['logging']['iteration_step_info_interval'],
        gradient_clip_val=config['optimizer']['clip_grad'],
        accumulate_grad_batches=config['tokens']['batch_accumulation_per_replica'],
        strategy='auto',
        enable_checkpointing=True,
        num_sanity_val_steps=0,
        max_epochs=None,
        check_val_every_n_epoch=None,
        enable_model_summary=True,
        reload_dataloaders_every_n_epochs=0,
        deterministic=False,
        detect_anomaly=False
    )

    # Add more detailed logging before training
    try:
        logging.info("-" * NUM_DASHES)
        logging.info("Starting training")
        logging.info(f"Total number of sequences: {len(dataset):,}")
        logging.info(f"Batch size: {train_loader.batch_size}")
        logging.info(f"Gradient accumulation steps: {trainer.accumulate_grad_batches}")
        logging.info(f"Effective batch size: {train_loader.batch_size * trainer.accumulate_grad_batches}")
        logging.info(f"Number of training steps: {config['tokens']['train_steps']}")
        logging.info(f"Learning rate: {model.learning_rate}")
        logging.info("-" * NUM_DASHES)
        
        # Clear CUDA cache before training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logging.info("Cleared CUDA cache before training")
        
        if resume_from_checkpoint:
            trainer.fit(
                model, 
                train_loader, 
                ckpt_path=resume_from_checkpoint
            )
        else:
            trainer.fit(model, train_loader)
            
        end_time = time.time()
        total_time = end_time - start_time

        logging.info("\nTraining completed successfully!")
        logging.info(f"Total training time: {timedelta(seconds=int(total_time))}")
        logging.info(f"Final step: {model.trainer.global_step}")
        logging.info(f"Checkpoints saved in: {checkpoint_callback.dirpath}")
        
    except Exception as e:
        logging.error(f"Training error occurred: {str(e)}", exc_info=True)
        raise e
    
    return model

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume_checkpoint', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--additional_steps', type=int, help='Number of additional steps to train')
    args = parser.parse_args()
    
    try:
        logging.info("Setting up environment variables")
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        os.environ['TORCH_USE_CUDA_DSA'] = '1'
        
        train_model(
            resume_from_checkpoint=args.resume_checkpoint,
            additional_steps=args.additional_steps
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Cleaning up...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        sys.exit(0)
    except Exception as e:
        logging.error(f"Error during training: {str(e)}", exc_info=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        sys.exit(1)

