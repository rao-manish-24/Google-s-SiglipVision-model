# Google's SiglipVision model
Comprehensive implementation of Google's SiglipVision model, built from scratch and validated against the Hugging Face implementation. SiglipVision is a vision transformer (ViT) model designed for image understanding, following the general architecture of vision transformers but with specific design choices.


# SiglipVision Model Implementation

A from-scratch PyTorch implementation of Google's SiglipVision model, with validation against the Hugging Face transformers implementation.

## Project Overview

This repository contains a complete implementation of the SiglipVision Transformer model, following the architecture described in the SigLIP (Sigmoid Loss for Language Image Pre-Training) paper. The implementation is built step-by-step, with each component validated against the Hugging Face transformers library implementation.

Key features:
- Complete implementation of SiglipVision model architecture
- Validation against Hugging Face's implementation
- Detailed explanation of each component
- Visualization of patch embeddings and model outputs

## Requirements

- Python 3.8+
- PyTorch 1.10+
- torchvision
- transformers
- Pillow
- matplotlib
- dataclasses

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/siglip-vision-implementation.git
cd siglip-vision-implementation

# Install dependencies
pip install -r requirements.txt
```

## Model Architecture

SiglipVision is a Vision Transformer (ViT) model with the following components:

1. **Vision Embeddings**
   - Divides input images into 16×16 non-overlapping patches
   - Projects patches to embedding dimension (768) using a convolutional layer
   - Adds learned positional embeddings

2. **Transformer Encoder**
   - 12 encoder layers
   - Each layer contains:
     - Pre-normalization (LayerNorm)
     - Multi-head self-attention (12 heads)
     - Residual connection
     - Another pre-normalization
     - MLP with GELU activation
     - Another residual connection

3. **Final Layer Normalization**
   - Applied to the output of the last encoder layer

## Usage

```python
import torch
from PIL import Image
from torchvision import transforms
from model import SiglipVisionModel, SiglipVisionConfig

# Load and preprocess image
image = Image.open("image.jpg")
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
image_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension

# Initialize model
config = SiglipVisionConfig(
    num_hidden_layers=12,
    hidden_size=768,
    intermediate_size=3072,
    num_attention_heads=12
)
model = SiglipVisionModel(config)

# For using pre-trained weights
# model.load_state_dict(torch.load("siglip_weights.pth"))

# Get image embeddings
with torch.no_grad():
    outputs = model(image_tensor)

# outputs shape: [1, 196, 768] (batch_size, num_patches, embedding_dim)
```

## Implementation Details

### Vision Embeddings

```python
class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Patch embedding with convolutional layer
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            padding="valid",
        )
        
        # Position embeddings
        self.num_patches = (config.image_size // config.patch_size) ** 2
        self.position_embedding = nn.Embedding(self.num_patches, config.hidden_size)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_patches).expand((1, -1)),
            persistent=False,
        )

    def forward(self, pixel_values):
        B, C, H, W = pixel_values.shape
        
        # Create patch embeddings
        patch_embeds = self.patch_embedding(pixel_values)
        embeddings = patch_embeds.flatten(start_dim=2, end_dim=-1)
        embeddings = embeddings.transpose(1, 2)
        
        # Add position embeddings
        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings
```

### Multi-Head Attention

```python
class SiglipAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.dropout = config.attention_dropout
        
        # Projection layers
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, hidden_states):
        B, T, C = hidden_states.shape
        
        # Project to query, key, value
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        q = q.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = k.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = v.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        
        # Scaled dot-product attention
        attn_weights = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        
        # Apply attention weights to values
        attn_output = attn_weights @ v
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).reshape(B, T, C)
        
        # Final projection
        attn_output = self.out_proj(attn_output)
        return attn_output
```

## Validation Against Hugging Face

The implementation includes validation steps to ensure compatibility with the Hugging Face transformers library:

```python
# Load Hugging Face model
from transformers import SiglipVisionModel as HFSiglipVisionModel
hf_model = HFSiglipVisionModel.from_pretrained("google/siglip-base-patch16-224")

# Load weights from Hugging Face model to our implementation
our_model = SiglipVisionModel(SiglipVisionConfig())
our_model.load_state_dict(hf_model.state_dict())

# Validate outputs
with torch.no_grad():
    our_output = our_model(image_tensor)
    hf_output = hf_model(image_tensor)
    max_diff = torch.max(torch.abs(our_output - hf_output))
    print(f"Max difference: {max_diff:.8f}")  # Should be very close to 0
```

## References

- [SigLIP: Signal Language Image Pre-training for Multi-modal Language Understanding](https://arxiv.org/abs/2303.15343)
- [Hugging Face Transformers Library](https://github.com/huggingface/transformers)
- [Vision Transformer (ViT)](https://arxiv.org/abs/2010.11929)

## License

MIT
