import torch
import torch.nn as nn
import torch.serialization
import torchvision.transforms as transforms
from PIL import Image

# Patch Embedding Layer
class PatchEmbeddingLayer(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, embedding_dim=768, img_size=224):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embedding_dim, kernel_size=patch_size, stride=patch_size)
        self.class_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        self.position_embeddings = nn.Parameter(torch.randn(1, self.num_patches + 1, embedding_dim))

    def forward(self, x):
        B = x.shape[0]
        x = self.proj(x).flatten(2).transpose(1, 2)
        cls_tokens = self.class_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.position_embeddings
        return x

# Multi-Head Self-Attention Block
class MultiHeadSelfAttentionBlock(nn.Module):
    def __init__(self, embedding_dims=768, num_heads=12, attn_dropout=0.0):
        super().__init__()
        self.layernorm = nn.LayerNorm(embedding_dims)
        self.multiheadattention = nn.MultiheadAttention(
            embed_dim=embedding_dims, 
            num_heads=num_heads,                                            
            dropout=attn_dropout, 
            batch_first=True
        )

    def forward(self, x):
        x_norm = self.layernorm(x)
        attn_output, _ = self.multiheadattention(x_norm, x_norm, x_norm, need_weights=False)
        return attn_output

# MLP Block
class MachineLearningPerceptronBlock(nn.Module):
    def __init__(self, embedding_dims=768, mlp_size=3072, mlp_dropout=0.1):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(embedding_dims, mlp_size),
            nn.GELU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(mlp_size, embedding_dims),
            nn.Dropout(mlp_dropout)
        )
        self.norm = nn.LayerNorm(embedding_dims)

    def forward(self, x):        
        x_norm = self.norm(x)
        return self.linear(x_norm)

# Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, embedding_dims=768, mlp_dropout=0.1, attn_dropout=0.0, mlp_size=3072, num_heads=12):
        super().__init__()
        self.msa_block = MultiHeadSelfAttentionBlock(embedding_dims, num_heads, attn_dropout)
        self.mlp_block = MachineLearningPerceptronBlock(embedding_dims, mlp_size, mlp_dropout)

    def forward(self, x):
        x = x + self.msa_block(x)
        x = x + self.mlp_block(x)
        return x

# Vision Transformer
class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, in_channels=3, patch_size=16, embedding_dims=768,
                 num_transformer_layers=12, mlp_dropout=0.1, attn_dropout=0.0, mlp_size=3072,
                 num_heads=12, num_classes=12):
        super().__init__()
        self.patch_embedding_layer = PatchEmbeddingLayer(in_channels, patch_size, embedding_dims, img_size)
        self.transformer_encoder = nn.Sequential(
            *[TransformerBlock(embedding_dims, mlp_dropout, attn_dropout, mlp_size, num_heads)
              for _ in range(num_transformer_layers)]
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(embedding_dims),
            nn.Linear(embedding_dims, num_classes)
        )

    def forward(self, x):
        x = self.patch_embedding_layer(x)
        x = self.transformer_encoder(x)
        x = x[:, 0]
        return self.classifier(x)

torch.serialization.add_safe_globals([VisionTransformer])

# Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# Load model weights
def load_model(weights_path="vit_weights.pt"):
    model = torch.load(weights_path, map_location="cpu", weights_only=False)
    model.eval()
    return model

# Predict from PIL image
def predict(model, image: Image.Image):
    x = transform(image).unsqueeze(0)  # (1, 3, 224, 224)
    with torch.no_grad():
        outputs = model(x)
        predicted_class = torch.argmax(outputs, dim=1).item()
    return predicted_class
