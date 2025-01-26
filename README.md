# SmolLM2 135M Language Model

## About the Project
SmolLM2 135M is a lightweight transformer-based language model designed for efficient training and inference on consumer hardware. It implements a modern decoder-only architecture with features like rotary positional embeddings, grouped-query attention, and RMSNorm for stable training.

## Features
- 135M parameter transformer model
- Grouped-Query Attention (GQA) for efficient attention computation
- RMSNorm for stable training
- Rotary Position Embeddings (RoPE)
- Weight tying between input embeddings and output layer
- Cosine learning rate schedule with warmup
- Gradient checkpointing support
- Automatic mixed precision training
- Configurable inference sampling

## Model Architecture & Parameter Calculations
The model uses the following architecture:
- Vocabulary Size: 49,152 tokens
- Hidden Size: 576
- Intermediate Size: 1,536
- Number of Layers: 30
- Attention Heads: 9
- Key/Value Heads: 3 (3x grouped-query attention)
- Max Sequence Length: 2,048
- Activation Function: SiLU (Swish)

Parameter count breakdown:
- Token Embeddings: 49,152 × 576 = 28,311,552
- Self-Attention (per layer):
  - Q/K/V Projections: 3 × (576 × 576) = 995,328
  - Output Projection: 576 × 576 = 331,776
- MLP (per layer):
  - Gate Projection: 576 × 1,536 = 884,736
  - Up Projection: 576 × 1,536 = 884,736
  - Down Projection: 1,536 × 576 = 884,736
- Layer Norm (per layer): 576 × 2 = 1,152

Total Parameters: ~135M

## Project Structure 
.  
├── train_model135m.py # Main training script  
├── model135m.py # Model architecture definition  
├── requirements.txt # Project dependencies  
└── lightning_checkpoints # Training checkpoints directory  

## Requirements
pytorch>=2.0.0
pytorch-lightning>=2.0.0
transformers>=4.30.0
tiktoken
datasets
tqdm

## Training Process & Configuration
The training process uses PyTorch Lightning with the following key configurations:

- Optimizer: AdamW
  - Learning Rate: 3e-4
  - Weight Decay: 0.1
  - Betas: (0.9, 0.999)
  - Epsilon: 1e-8

- Training Parameters:
  - Batch Size: 2
  - Gradient Accumulation Steps: 8
  - Effective Batch Size: 16
  - Max Steps: 5000
  - Warmup Steps: 100
  - Gradient Clipping: 1.0
  - Precision: 32-bit

- Checkpointing:
  - Save Every: 20 steps
  - Keep Top: 5 checkpoints
  - Monitor: train_loss

## Initial training (v2 is non pytorch-lightning version)
python train_model135m.py --total_steps 5000  
python train_model135m_v2.py --total_steps 5000
 
## Resume training with additional steps (v2 is non pytorch-lightning version)
python train_model135m.py --additional_steps 500  
python train_model135m_v2.py --additional_steps 500

## Training Logs

Sample training progress:
Log 1
![Training Log 1](./images/training-scr1.png?raw=true "Training Log 1")

Log 2:
![Training Log 2](./images/training-scr2.png?raw=true "Training Log 2")

Log 3:
![Training Log 2](./images/training-scr3.png?raw=true "Training Log 3")

Log 4:
![Training Log 2](./images/training-scr4.png?raw=true "Training Log 4")
