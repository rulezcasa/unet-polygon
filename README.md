# Polygon-UNet: Conditional Image Colorization

- This repository is the codebase for a simple Unet trained from scratch to colour polygon images. Submitted as an screening assignment for Ayna. 
- Script files have been appropriately commented to demonstrate the overall logic and flow.
- Curious to build from scratch, I didnt try out the Hugging Face Diffuser API. It could've been a more stable setup.

## Overview

- **Input**: Grayscale polygon images (triangles, squares, circles, etc.)
- **Condition**: Color text prompt (e.g., "red", "blue", "green")
- **Output**: Colorized polygon images with the specified color

The model uses color embeddings to condition the generation process, allowing it to produce different colored outputs for the same input shape.

## 📁 Directory Structure

```
Polygon-Unet/
├── train_best.py               # Main training script with the best model config
├── model.py                    # Conditional U-Net architecture implementation
├── dataset.py                  # Custom dataset loader for polygon images
├── inference.ipynb             # Interactive inference notebook
├── requirements.txt            # Python dependencies
├── dataset/                    # Training and validation data
│   ├── training/
│   │   ├── inputs/             # Input polygon images
│   │   ├── outputs/            # Target colored images
│   │   └── data.json           # Metadata mapping inputs to outputs
│   └── validation/
│       ├── inputs/
│       ├── outputs/
│       └── data.json
├── hyperparameter_tuning/       
│   ├── grid_search.py              # multiprocesed grid search pipeline for parameter tuning
│   └── grid_search_summary.txt     # Summary of parameters
|
├── best_model/                     #Best model
|
└──── models/                       # Saved models
```

## Architecture

- **Encoder**: Compresses input images through downsampling layers
- **Decoder**: Reconstructs images through upsampling with skip connections
- **Color Conditioning**: Embeds color text into learnable vectors and concatenates with input
- **Skip Connections**: Preserves spatial information between encoder and decoder

### Key Components
- **Color Embedding**: Converts color text to dense vectors
- **DoubleConv Blocks**: Standard 3x3 convolutions with BatchNorm and ReLU
- **Up/Down Blocks**: Handles resolution changes with optional bilinear upsampling

### My Intution of a Unet:
- The encoder is like squinting at the image — it shrinks things down to understand the big picture.
- The decoder zooms back in, rebuilding the image step by step.
- Skip connections are like bookmarks and help the model remember fine details while zooming back in.
- We turn the color name into a small vector (like a color vibe) and feed it to the model so it knows what shade to paint.
- Each block just stacks a couple of convolutions with ReLU and BatchNorm to refine the features.


## Hyperparameter Tuning

An automated grid search for hyperparameter optimization was employed resulting in 24 sweeps trained for 15 epochs using a multiprocessing pipeline:

### Grid Search Parameters
```python
# Learning Rates
lrs = [1e-4, 3e-4, 1e-3]

# Loss Functions
losses = ["L1Loss", "MSELoss"]

# Embedding Dimensions
embed_dims = [8, 16]

# Upsampling Methods
bilinears = [True, False]  # Bilinear vs Transposed Convolution
```

### Grid Search Configuration
- **Total Combinations**: 24 different configurations
- **Training Epochs**: 15 epochs per configuration 
- **Batch Size**: 8
- **Optimizer**: Adam
- **Parallel Processing**: Up to 4 workers
- **Run name**: Run_{Run_number}_{lr}_{loss_type}_{dim}_billinear{True/False}

### Best selected hyperparameters
- Selected based on loss graphs and training stability (R23) and trained for 50 epochs (Noticed a loss plateau around this point).

```python
    # Best config
    lr = 0.001
    loss_name = "MSELoss"
    embed_dim = 16
    bilinear = True
    run_name = "Final_R23_best_lr0.001_lossMSELoss_dim16_bilinearTrue"
```

## Run Inference
Open `inference.ipynb` and follow the interactive prompts to:
- Load the best model
- Input the path of a polygon image
- Specify a color
- Generate the colored output

### Supported Colors
- `red`, `green`, `blue`, `yellow`, `orange`
- `purple`, `pink`, `black`, `white`, `cyan`, `magenta`

### Graphs and outputs:

## Imporvements for later 

### Would help :
- **Residual Connections**: Add residual connections throughout the network for better gradient flow 
- **Progressive Training**: Start with low-resolution images and gradually increase resolution

### Might be an overkill since the task is simple
- **GAN Integration**: Incorporate adversarial training for more realistic colorization results
- **Attention Mechanisms**: Implement self-attention or cross-attention layers to better capture spatial relationships

## AI usage
- To understand and revise my understanding of the Unet architecture.
- To setup multiprocessing pipeline for grid search.
- Some level of codebase structuring and formatting.