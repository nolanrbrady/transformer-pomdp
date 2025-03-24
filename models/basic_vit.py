import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PatchEmbedding(nn.Module):
    """
    Image to Patch Embedding.
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768, pad_if_needed=True):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.pad_if_needed = pad_if_needed
        
        # Calculate dimensions
        if isinstance(img_size, int):
            self.h = self.w = img_size
        else:
            self.h, self.w = img_size
            
        # Calculate number of patches
        self.h_patches = self.h // patch_size
        self.w_patches = self.w // patch_size
        self.num_patches = self.h_patches * self.w_patches
        
        # Projection layer: conv is equivalent to splitting image into patches + linear projection
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Class token and position embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        
        # Initialize parameters
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
    def forward(self, x):
        """
        x: (B, C, H, W) - Batch of images
        """
        B, C, H, W = x.shape
        
        # Add padding if needed
        pad_h = pad_w = 0
        if self.pad_if_needed:
            pad_h = (self.patch_size - H % self.patch_size) % self.patch_size
            pad_w = (self.patch_size - W % self.patch_size) % self.patch_size
            
            if pad_h > 0 or pad_w > 0:
                x = F.pad(x, (0, pad_w, 0, pad_h))
        
        # Project patches
        x = self.proj(x)  # (B, embed_dim, grid_h, grid_w)
        x = x.flatten(2)  # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (B, num_patches, embed_dim)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, 1 + num_patches, embed_dim)
        
        # Add position embeddings
        x = x + self.pos_embed
        return x

class MultiHeadAttention(nn.Module):
    """
    Multi-head Attention module.
    """
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B, N, C = x.shape
        
        # Linear projection to get Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to V and reshape
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        
        return x

class MLP(nn.Module):
    """
    MLP module with GELU activation.
    """
    def __init__(self, in_features, hidden_features, out_features, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class TransformerBlock(nn.Module):
    """
    Transformer Block: Multi-Head Attention + MLP with LayerNorm.
    """
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(
            in_features=embed_dim,
            hidden_features=int(embed_dim * mlp_ratio),
            out_features=embed_dim,
            dropout=dropout,
        )
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class BasicViT(nn.Module):
    """
    Basic Vision Transformer for reinforcement learning tasks.
    Takes a single image input and outputs action logits.
    
    Args:
        img_size (int or tuple): Input image size (height, width).
        patch_size (int): Patch size.
        in_channels (int): Number of input channels.
        num_classes (int): Number of output classes (actions).
        embed_dim (int): Embedding dimension.
        depth (int): Number of transformer blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        dropout (float): Dropout rate.
        pad_if_needed (bool): Whether to pad input images.
    """
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=1000,
        embed_dim=768,
        depth=6,
        num_heads=12,
        mlp_ratio=4.0,
        dropout=0.0,
        pad_if_needed=True,
    ):
        super().__init__()
        
        # Save parameters
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        
        # Patch + Position Embedding
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            pad_if_needed=pad_if_needed,
        )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(depth)
        ])
        
        # Layer Norm
        self.norm = nn.LayerNorm(embed_dim)
        
        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
    
    def forward(self, x):
        """
        Forward pass for a batch of images.
        
        Args:
            x: Images of shape (B, C, H, W)
            
        Returns:
            logits: Class logits of shape (B, num_classes)
        """
        # Make sure input is a tensor
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(np.array(x), dtype=torch.float32)
            
        # Handle single image case by adding batch dimension
        if len(x.shape) == 3:  # (C, H, W)
            x = x.unsqueeze(0)  # Add batch dimension: (1, C, H, W)
            
        # Patch embedding
        x = self.patch_embed(x)  # (B, n_patches+1, embed_dim)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Apply layer norm
        x = self.norm(x)
        
        # Use CLS token for classification
        x = x[:, 0]  # (B, embed_dim)
        
        # Classification head
        x = self.head(x)  # (B, num_classes)
        
        return x
    
    def act(self, obs):
        """
        Given an observation, sample an action.
        
        Args:
            obs: Single image of shape (C, H, W) or batch (B, C, H, W)
            
        Returns:
            action: Index of the selected action
            log_prob: Log probability of the selected action
        """
        # Make sure input is a tensor
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(np.array(obs), dtype=torch.float32)
            
        # Normalize if not already done
        if obs.max() > 1.0:
            obs = obs / 255.0
            
        # Get action logits
        logits = self.forward(obs)  # (B, num_classes)
        
        # For batch size 1, remove batch dimension
        if logits.shape[0] == 1:
            logits = logits.squeeze(0)  # (num_classes)
        
        # Compute action probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Create categorical distribution
        dist = torch.distributions.Categorical(probs)
        
        # Sample action
        action = dist.sample()
        
        # Get log probability
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob 