import os
import torch
from torch.utils.data import Dataset, DataLoader
from model135m_v2 import SmolLM2ForCausalLM135M, SmolLM2Config135M
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from typing import Optional, List
import logging
from tqdm.auto import tqdm
import time
from datetime import timedelta
from datasets import load_dataset
import glob
import re
from torch.cuda.amp import autocast, GradScaler
from torch.nn.functional import scaled_dot_product_attention

# Constants 
BATCH_SIZE = 4  # Reduced from 16
SEQUENCE_LENGTH = 512  # Reduced from 1024
CHECKPOINT_SAVE_INTERVAL = 500  # Less frequent saving
SAMPLE_GENERATION_INTERVAL = 500  # Less frequent sampling

# Constants
TOTAL_STEPS = 5000
DATA_ELEMENTS = 2000
NUM_DASHES = 100
MIN_TOKENS = 10
MAX_TOKENS = 2048
VOCAB_SIZE = 50304
DEVICE = "cpu"  # Force CPU usage

# Update constants
LEARNING_RATE = 6e-4  
WARMUP_STEPS = 2000
GRAD_ACCUM_STEPS = 1  
GRAD_CLIP = 1.0  

# Remove unnecessary validations and checks
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class StreamingDataset:
    def __init__(self, block_size=1024):
        self.block_size = block_size
        self.vocab_size = 50304  
        
    def __getitem__(self, idx):
        # Fast random data generation 
        x = torch.randint(0, self.vocab_size, (self.block_size,))
        y = torch.randint(0, self.vocab_size, (self.block_size,))
        return x, y
    
    def __len__(self):
        return 10000  # Large enough number

def collate_batch(batch):
    """Custom collate function to pad sequences in a batch."""
    # Find max length in the batch
    max_len = max(x[0].size(0) for x in batch)
    
    # Initialize tensors for inputs and targets
    batch_size = len(batch)
    inputs = torch.full((batch_size, max_len), fill_value=0, dtype=torch.long)  # 0 is assumed to be pad_token_id
    targets = torch.full((batch_size, max_len), fill_value=-100, dtype=torch.long)  # -100 is ignored by CrossEntropyLoss
    
    # Fill in the tensors with actual values
    for i, (input_seq, target_seq) in enumerate(batch):
        seq_len = input_seq.size(0)
        inputs[i, :seq_len] = input_seq
        targets[i, :seq_len] = target_seq
    
    return inputs, targets

def save_checkpoint(model, optimizer, scheduler, step, loss, path):
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
    }, path)

def load_checkpoint(path, model, optimizer=None, scheduler=None):

    logging.info(f"Loading checkpoint from {path}, Please wait...")
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint['step'], checkpoint['loss']

def get_latest_checkpoint() -> tuple[Optional[str], Optional[int]]:
    """Returns tuple of (checkpoint_path, step_number) or (None, None) if no checkpoints exist"""
    checkpoints = glob.glob("checkpoints/*.pt")
    if not checkpoints:
        return None, None
    
    # Print all checkpoints for debugging
    print("\nFound checkpoints:")
    for ckpt in checkpoints:
        print(f"  {ckpt}")
    
    steps = []
    for ckpt in checkpoints:
        # Extract step number from filename
        match = re.search(r'model_step_(\d+)_loss_', ckpt)
        if match:
            step_num = int(match.group(1))
            steps.append((step_num, ckpt))
            print(f"  Extracted step {step_num} from {ckpt}")
    
    if steps:
        latest_step, latest_ckpt = max(steps, key=lambda x: x[0])
        print(f"\nSelected checkpoint:")
        print(f"  Path: {latest_ckpt}")
        print(f"  Step: {latest_step}\n")
        return latest_ckpt, latest_step
    return None, None

def generate_sample(model, tokenizer, device, prompt="Once upon a time"):
    model.eval()
    try:
        with torch.no_grad():
            # Tokenize with proper padding
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=model.config.max_position_embeddings,
                return_attention_mask=True,
                add_special_tokens=True
            )
            
            # Move to device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=100,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                num_return_sequences=1,
                no_repeat_ngram_size=2
            )
            
            # Decode
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
        model.train()
        return generated_text
        
    except Exception as e:
        logging.error(f"Error during text generation: {str(e)}")
        model.train()
        return f"Error generating text: {str(e)}"

class DataLoaderLite:
    """Simple data loader"""
    def __init__(self, B=4, T=512):
        self.B = B
        self.T = T
        
    def next_batch(self):
        x = torch.randint(0, VOCAB_SIZE, (self.B, self.T))
        y = torch.randint(0, VOCAB_SIZE, (self.B, self.T))
        return x, y

def train_model(total_steps: int = TOTAL_STEPS, additional_steps: Optional[int] = None):
    try:
        # Setup device
        logging.info("Setting up device, Please wait...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.backends.cudnn.benchmark = True
        
        # Create checkpoints directory
        logging.info("Creating checkpoints directory, Please wait...")
        os.makedirs("checkpoints", exist_ok=True)
        
        # Initialize model with smaller config
        logging.info("Initializing model with smaller config, Please wait...")
        config = SmolLM2Config135M(
            vocab_size=VOCAB_SIZE,
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=6,
            num_attention_heads=4,
            num_key_value_heads=2,
            hidden_act="silu",
            max_position_embeddings=512,
            initializer_range=0.041666666666666664,
            rms_norm_eps=1.0e-05
        )
        
        model = SmolLM2ForCausalLM135M(config).to(device)
        
        # Initialize tokenizer with padding token
        logging.info("Initializing tokenizer with padding token, Please wait...")
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/cosmo2-tokenizer")
        # Add padding token if it doesn't exist
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # Also ensure other special tokens are set
        if tokenizer.bos_token is None:
            tokenizer.bos_token = tokenizer.eos_token
            tokenizer.bos_token_id = tokenizer.eos_token_id
            
        # Update model config with tokenizer's special tokens
        config.pad_token_id = tokenizer.pad_token_id
        config.bos_token_id = tokenizer.bos_token_id
        config.eos_token_id = tokenizer.eos_token_id
        
        # Initialize data loader
        logging.info("Initializing data loader, Please wait...")
        train_loader = DataLoaderLite(B=BATCH_SIZE, T=SEQUENCE_LENGTH)
        
        # Initialize optimizer and scheduler
        logging.info("Initializing optimizer and scheduler, Please wait...")
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            betas=(0.9, 0.95),
            weight_decay=0.1,
            fused=True
        )
        
        # Determine start step and total steps
        start_step = 0
        if additional_steps is not None:
            # Try to load the latest checkpoint
            latest_ckpt, latest_step = get_latest_checkpoint()
            if latest_ckpt is not None:
                print(f"\nResuming from checkpoint: {latest_ckpt}")
                print(f"Last completed step: {latest_step}")
                checkpoint = torch.load(latest_ckpt)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_step = latest_step + 1
                total_steps = start_step + additional_steps
                print(f"Will train from step {start_step} to {total_steps}\n")
            else:
                print("No checkpoint found. Starting from step 0.")
                total_steps = additional_steps
        
        # Initialize scheduler with correct number of steps
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=WARMUP_STEPS,
            num_training_steps=total_steps
        )
        
        # Load scheduler state if resuming
        if additional_steps is not None and latest_ckpt is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Training loop
        logging.info("Starting training loop, Please wait...")
        model.train()
        for step in range(start_step, total_steps):
            t0 = time.time()
            
            # Get batch directly
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            
            # Forward pass
            outputs = model(input_ids=x, labels=y)
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            
            # Log progress
            if step % 10 == 0:
                dt = time.time() - t0
                tokens_per_sec = (BATCH_SIZE * SEQUENCE_LENGTH) / dt
                lr = scheduler.get_last_lr()[0]
                print(f'Step {step} | Loss: {loss.item():.4f} | lr: {lr:.2e} | dt: {dt*1000:.2f}ms | tok/s: {tokens_per_sec:.2f}')
            
            # Generate sample text every 500 steps
            if step > 0 and (step % SAMPLE_GENERATION_INTERVAL == 0 or step == total_steps - 1):
                print(f"\n{'='*NUM_DASHES}")
                print(f"Generating text sample at step {step}")
                print(f"{'-'*NUM_DASHES}")
                sample_text = generate_sample(model, tokenizer, device)
                print(f"Generated text:\n{sample_text}")
                print(f"{'-'*NUM_DASHES}\n")
                model.train()  # Ensure model is back in training mode
            
            # Save checkpoint every 500 steps and at the final step
            if step > 0 and (step % CHECKPOINT_SAVE_INTERVAL == 0 or step == total_steps - 1):
                # Format loss for filename (e.g., 2.345 becomes "2_34")
                loss_str = f"{loss.item():.2f}".replace('.', '_')
                checkpoint_path = f"checkpoints/model_step_{step}_loss_{loss_str}.pt"
                print(f"Saving checkpoint to {checkpoint_path}")
                torch.save({
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': loss.item(),
                    'config': config,
                }, checkpoint_path)
                print(f"Checkpoint saved successfully\n")
                
    except Exception as e:
        logging.error(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--total_steps', type=int, default=TOTAL_STEPS,
                      help='Total number of steps for fresh training')
    parser.add_argument('--additional_steps', type=int,
                      help='Number of additional steps when resuming training')
    args = parser.parse_args()
    
    if args.additional_steps is not None:
        train_model(additional_steps=args.additional_steps)
    else:
        train_model(total_steps=args.total_steps) 

    logging.info("Training completed successfully !!!")