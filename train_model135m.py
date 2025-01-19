import os
import math
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar, TQDMProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
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
        
        # Load and tokenize text
        logging.info("Loading text from input.txt...")
        with open('input.txt', 'r') as f:
            text = f.read()
        logging.info(f"Loaded text with {len(text)} characters")
        
        # Initialize tokenizer
        logging.info("-" * NUM_DASHES)
        logging.info("Initializing tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/cosmo2-tokenizer")
        logging.info("Tokenizing text...")
        tokens = self.tokenizer.encode(text)
        logging.info(f"Initial tokenization complete: {len(tokens)} tokens")
        
        # Token processing
        logging.info("Processing tokens...")
        tokens = [t if t < vocab_size else self.tokenizer.unk_token_id for t in tokens]
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        
        # Calculate statistics
        self.n_samples = max(0, (len(self.tokens) - self.block_size - 1))
        logging.info("-" * NUM_DASHES)
        logging.info(f"Dataset statistics:")
        logging.info(f"- Total tokens: {len(self.tokens)}")
        logging.info(f"- Block size: {self.block_size}")
        logging.info(f"- Number of samples: {self.n_samples}")
        
        # Validate token range
        max_token = self.tokens.max().item()
        min_token = self.tokens.min().item()
        logging.info(f"Token range validation:")
        logging.info(f"- Minimum token ID: {min_token}")
        logging.info(f"- Maximum token ID: {max_token}")
        logging.info(f"- Vocabulary size: {vocab_size}")
        assert max_token < vocab_size, f"Found token {max_token} >= vocab_size {vocab_size}"
        
        logging.info("Dataset initialization complete")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        if idx >= self.n_samples:
            raise IndexError(f"Index {idx} out of bounds for dataset with {self.n_samples} samples")
        
        # Get sequence
        chunk = self.tokens[idx:idx + self.block_size + 1].clone()
        
        # Validate sequence
        assert chunk.max() < self.vocab_size, f"Token value {chunk.max()} exceeds vocab size {self.vocab_size}"
        assert len(chunk) == self.block_size + 1, f"Sequence length {len(chunk)} != {self.block_size + 1}"
        
        x = chunk[:-1]
        y = chunk[1:]
        return x, y

class SmolLM2LightningModule(pl.LightningModule):
    def __init__(
        self, 
        config: Optional[SmolLM2Config135M] = None,
        learning_rate: float = 3e-4,
        total_steps: int = 5000
    ):
        super().__init__()
        
        # Store parameters as instance variables
        self.learning_rate = learning_rate
        self.total_steps = total_steps
        
        # Remove autocast initialization as it's handled by Lightning
        self.save_hyperparameters()

        logging.info("-" * NUM_DASHES)
        logging.info("Initializing SmolLM2LightningModule")
        logging.info(f"- Learning rate: {learning_rate}")
        logging.info(f"- Total steps: {total_steps}")
        
        # Initialize tokenizer first
        # logging.info("Initializing tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/cosmo2-tokenizer")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # logging.info("Tokenizer initialization complete")
        
        # Initialize model and config
        self.config = config if config else SmolLM2Config135M()
        showModelConfig = False
        if showModelConfig:
            logging.info("Model configuration:")
            for key, value in self.config.__dict__.items():
                if not key.startswith('_'):
                    logging.info(f"- {key}: {value}")
        
        logging.info("-" * NUM_DASHES)
        logging.info("Initializing model...")
        self.model = SmolLM2ForCausalLM135M(self.config)
        logging.info("Model initialization complete")
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logging.info(f"Model parameters:")
        logging.info(f"- Total: {total_params:,}")
        logging.info(f"- Trainable: {trainable_params:,}")
        
        self.training_start_time = None
        self.last_log_time = None
        self.step_times = []
        
        # Add prompt for inference
        self.test_prompt = "Once upon a time"
        self.inference_steps = 500
        
        # Add step tracking
        self.automatic_optimization = True
        
    def generate_sample_text(self):
        """Generate sample text using the current model state"""
        logging.info("\nGenerating sample text...")
        logging.info(f"Prompt: {self.test_prompt}")
        
        # Tokenize the prompt with padding
        inputs = self.tokenizer(
            self.test_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_position_embeddings,
            return_attention_mask=True
        )
        
        # Move inputs to device
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
        
        try:
            # Generate text
            with torch.no_grad():
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
            
        except Exception as e:
            logging.error(f"Error during text generation: {str(e)}")
            return None

    def training_step(self, batch, batch_idx):
        if self.training_start_time is None:
            self.training_start_time = time.time()
            self.last_log_time = self.training_start_time
        
        step_start = time.time()
        x, y = batch
        
        # Forward pass
        outputs = self.model(input_ids=x, labels=y)
        loss = outputs.loss
        
        # Calculate training statistics
        step_end = time.time()
        step_time = step_end - step_start
        self.step_times.append(step_time)
        
        # Log metrics for progress bar
        self.log_dict({
            'train_loss': loss,
            'learning_rate': self.learning_rate,
            'step': self.global_step,
            'step_time': step_time
        }, prog_bar=True, sync_dist=True)
        
        # Existing logging code
        if self.global_step % 50 == 0:
            current_time = time.time()
            elapsed = current_time - self.training_start_time
            time_per_step = sum(self.step_times[-100:]) / min(len(self.step_times), 100)
            steps_remaining = self.total_steps - self.global_step
            eta = steps_remaining * time_per_step
            
            logging.info(f"\nBatch {batch_idx}: Input shape: {x.shape}, Input device: {x.device}, "
                        f"Input range: [{x.min().item()}, {x.max().item()}]")
            if torch.cuda.is_available():
                logging.info(f"GPU Memory: {torch.cuda.memory_allocated() / 1024**2:.1f}MB allocated, "
                            f"{torch.cuda.memory_reserved() / 1024**2:.1f}MB reserved")
            
            logging.info(f"\nTraining Status - Step {self.global_step}/{self.total_steps}")
            logging.info(f"- Loss: {loss.item():.4f}, Time elapsed: {timedelta(seconds=int(elapsed))}, "
                        f"Average step time: {time_per_step:.3f}s, "
                        f"Estimated time remaining: {timedelta(seconds=int(eta))}")
        
        return loss

    def on_train_start(self):
        """Generate initial sample before training starts"""
        logging.info("\nGenerating initial sample before training:")
        self.generate_sample_text()

    def on_train_end(self):
        """Generate final sample after training completes"""
        logging.info("\nGenerating final sample after training:")
        self.generate_sample_text()

    def configure_optimizers(self):
        logging.info(f"Configuring optimizer with learning rate: {self.learning_rate}")
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        return optimizer

def train_model(
    resume_from_checkpoint: Optional[str] = None,
    additional_steps: Optional[int] = None
):
    start_time = time.time()
    logging.info("=" * NUM_DASHES)
    logging.info("Starting training process ...")
    logging.info("=" * NUM_DASHES)
    
    # Setup training environment (will run only once)
    setup_training_environment()
    
    # Set up seeds
    pl.seed_everything(1566)
    logging.info("Random seeds set")
    
    # Initialize config and dataset
    # logging.info("Initializing model configuration")
    config = SmolLM2Config135M()

    logging.info("-" * NUM_DASHES)    
    logging.info("Setting up dataset and data loader")
    dataset = TextDataset(block_size=128, vocab_size=config.vocab_size)
    
    # Worker setup
    num_workers = min(15, os.cpu_count() or 1)
    logging.info(f"Using {num_workers} workers for data loading")
    
    train_loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=True,
        prefetch_factor=2,
    )
    
    # Update step calculation logic
    if resume_from_checkpoint and additional_steps:
        logging.info(f"Loading checkpoint: {resume_from_checkpoint}")
        # Load checkpoint to get current step
        checkpoint = torch.load(resume_from_checkpoint)
        # Get the global step from checkpoint
        current_step = checkpoint.get('global_step', 0)
        logging.info(f"Resuming from global step {current_step}")
        # Calculate total steps as current + additional
        total_steps = current_step + additional_steps
        logging.info(f"Will train for {additional_steps} more steps until step {total_steps}")
    else:
        total_steps = 500  # Default steps for new training
        current_step = 0
        logging.info(f"Starting new training for {total_steps} steps")
    
    # Model setup
    # logging.info("Initializing model")
    model = SmolLM2LightningModule(config=config, total_steps=total_steps)
    
    # Create progress bar callback with simplified configuration
    progress_bar = RichProgressBar(
        refresh_rate=1, 
        leave=False, 
        theme=RichProgressBarTheme(
            description='', 
            progress_bar='#6206E0', 
            progress_bar_finished='#6206E0', 
            progress_bar_pulse='#6206E0', 
            batch_progress='', 
            time='dim', 
            processing_speed='dim underline', 
            metrics='italic', 
            metrics_text_delimiter=' ', 
            metrics_format='.3f'), 
        console_kwargs=None
    )
    
    # Update checkpoint callback to save global step
    checkpoint_callback = ModelCheckpoint(
        dirpath='lightning_checkpoints',
        filename='smollm2-{step}-{train_loss:.2f}',
        every_n_train_steps=20,
        save_top_k=-1,
        save_last=True,
        save_on_train_epoch_end=False
    )
    
    # Update trainer setup with RichProgressBar
    trainer = pl.Trainer(
        max_steps=total_steps,
        callbacks=[checkpoint_callback, progress_bar],
        accelerator='auto',
        devices=1,
        precision="16-mixed",
        enable_progress_bar=True,
        logger=True,
        log_every_n_steps=1,
        gradient_clip_val=1.0,
        accumulate_grad_batches=16,
        strategy='auto',
        enable_checkpointing=True,
        reload_dataloaders_every_n_epochs=1,
        num_sanity_val_steps=0,
        max_epochs=None,
        check_val_every_n_epoch=None
    )
    
    # Training
    try:
        logging.info("-" * NUM_DASHES)
        logging.info("Starting training")
        if resume_from_checkpoint:
            # Pass global_step to ensure correct step counting
            trainer.fit(
                model, 
                train_loader, 
                ckpt_path=resume_from_checkpoint
            )
        else:
            trainer.fit(model, train_loader)
            
        end_time = time.time()
        total_time = end_time - start_time

        logging.info("-" * NUM_DASHES)
        logging.info("\nTraining completed successfully!")
        logging.info(f"Total training time: {timedelta(seconds=int(total_time))}")
        logging.info(f"Final step: {model.global_step}")
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
    
    logging.info("Setting up environment variables")
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['TORCH_USE_CUDA_DSA'] = '1'
    
    # logging.info("Starting training script ...")
    train_model(
        resume_from_checkpoint=args.resume_checkpoint,
        additional_steps=args.additional_steps
    )

