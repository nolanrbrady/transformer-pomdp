import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PatchEmbedding(nn.Module):
    """
    Image to Patch Embedding with position embeddings.
    Now supports both single-image and batched inputs.
    """
    def __init__(self, img_size=84, patch_size=14, in_channels=3, embed_dim=768, pad_if_needed=True):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.pad_if_needed = pad_if_needed
        
        # Calculate target dimensions that are divisible by patch_size
        if isinstance(img_size, int):
            self.target_height = self.target_width = img_size
            self.h_patches = self.w_patches = img_size // patch_size
        else:
            self.target_height, self.target_width = img_size
            self.h_patches = self.target_height // patch_size
            self.w_patches = self.target_width // patch_size
            
        self.n_patches = self.h_patches * self.w_patches
        
        # Convolution used for patch projection
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Class token and position embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches + 1, embed_dim))
        
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
    
    def forward(self, x):
        """
        Expects x with shape (B, C, H, W) or (C, H, W).
        """
        if x.dim() == 3:
            x = x.unsqueeze(0)
        B, C, H, W = x.shape
        
        # Pad if necessary so dimensions are divisible by patch_size
        if self.pad_if_needed:
            pad_h = (self.patch_size - H % self.patch_size) % self.patch_size
            pad_w = (self.patch_size - W % self.patch_size) % self.patch_size
            if pad_h > 0 or pad_w > 0:
                x = F.pad(x, (0, pad_w, 0, pad_h))
        
        # Create patches and project them
        x = self.proj(x)  # (B, embed_dim, grid_h, grid_w)
        grid_h, grid_w = x.shape[-2], x.shape[-1]
        x = x.flatten(2)              # (B, embed_dim, n_patches)
        x = x.transpose(1, 2)         # (B, n_patches, embed_dim)
        
        # Add class token to each element in the batch
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)           # (B, n_patches+1, embed_dim)
        
        # If number of patches has changed (due to padding), interpolate position embeddings
        if x.size(1) != self.pos_embed.size(1):
            cls_pos_embed = self.pos_embed[:, 0:1, :]
            patch_pos_embed = self.pos_embed[:, 1:, :]
            
            n_h = (H + pad_h) // self.patch_size
            n_w = (W + pad_w) // self.patch_size
            
            patch_pos_embed = patch_pos_embed.reshape(1, self.h_patches, self.w_patches, -1)
            patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
            patch_pos_embed = F.interpolate(patch_pos_embed, size=(n_h, n_w), mode='bilinear')
            patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).flatten(1, 2)
            pos_embed = torch.cat([cls_pos_embed, patch_pos_embed], dim=1)
            x = x + pos_embed
        else:
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
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
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

class Block(nn.Module):
    """
    Transformer Block: Multi-Head Attention + MLP with LayerNorm.
    """
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), embed_dim, dropout)
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class InfiniAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, memory_size=64, window_size=8, dropout=0.1, ema_decay=0.9):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.memory_size = memory_size
        self.window_size = window_size
        self.head_dim = embed_dim // num_heads
        self.ema_decay = ema_decay  # Exponential moving average decay
        self.gate_param = nn.Parameter(torch.zeros(1))
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Learned compressive projection for memory
        self.memory_proj = nn.Linear(embed_dim, embed_dim)
        self.register_buffer("memory", torch.zeros(1, memory_size, embed_dim, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
        self.register_buffer("memory_initialized", torch.tensor(False, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
    
    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv  # (B, heads, T, head_dim)
        
        # Local attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_scores, dim=-1)
        local_out = torch.matmul(attn_weights, v)
        
        # Memory attention
        mem = self.memory_proj(self.memory)  # (1, M, D)
        mem = mem.expand(B, -1, -1)
        mem_kv = mem.reshape(B, self.memory_size, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        mem_attn = torch.matmul(q, mem_kv.transpose(-2, -1)) / np.sqrt(self.head_dim)
        mem_attn_weights = F.softmax(mem_attn, dim=-1)
        mem_out = torch.matmul(mem_attn_weights, mem_kv)
        
        # Combine local and memory outputs with a trainable gate
        gate = torch.sigmoid(self.gate_param)
        out = gate * local_out + (1 - gate) * mem_out
        out = out.transpose(1, 2).reshape(B, T, C)
        out = self.out_proj(out)
        
        # EMA Memory Update
        new_summary = x[:, -self.window_size:, :].mean(dim=1, keepdim=True)
        if not self.memory_initialized.item():
            self.memory.copy_(new_summary.mean(dim=0, keepdim=True).repeat(1, self.memory_size, 1).detach())
            self.memory_initialized.fill_(True)
        else:
            ema_updated = self.ema_decay * self.memory[:, -1:, :] + (1 - self.ema_decay) * new_summary.mean(dim=0, keepdim=True)
            self.memory = torch.cat([self.memory[:, 1:], ema_updated.detach()], dim=1)
        
        return out

class TemporalBlock(nn.Module):
    """
    Temporal Transformer Block using InfiniAttention and MLP with LayerNorm.
    """
    def __init__(self, embed_dim, num_heads, memory_size=64, window_size=8, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = InfiniAttention(embed_dim, num_heads, memory_size=memory_size, window_size=window_size, dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), embed_dim, dropout)
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class InfiniViT(nn.Module):
    """
    Infini Vision Transformer with spatial and temporal transformers,
    optimized for batched image preprocessing.
    """
    def __init__(
        self,
        img_size=84,
        patch_size=14,
        in_channels=3,
        num_classes=1000,
        embed_dim=1024,
        num_heads=12,
        memory_size=64,
        window_size=8,
        mlp_ratio=4.0,
        dropout=0.0,
        pad_if_needed=True,
        device=None,
        num_spatial_blocks=3,
        num_temporal_blocks=3,
    ):
        super().__init__()
        
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.memory_size = memory_size
        self.window_size = window_size
        
        # Determine target size based on img_size type
        if isinstance(img_size, int):
            h_patches = w_patches = img_size // patch_size
            target_size = (img_size, img_size)
        else:
            h_patches = img_size[0] // patch_size
            w_patches = img_size[1] // patch_size
            target_size = img_size
            
        self.n_patches = h_patches * w_patches
        print(f"Using grid of {h_patches}x{w_patches} patches = {self.n_patches} total patches")
        
        # Patch and position embedding
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            pad_if_needed=pad_if_needed,
        )
        
        # Spatial transformer blocks and norms
        self.spatial_blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_spatial_blocks)
        ])
        self.spatial_norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_spatial_blocks + 1)])
        
        # Temporal transformer blocks and norms
        self.temporal_blocks = nn.ModuleList([
            TemporalBlock(embed_dim, num_heads, mlp_ratio=mlp_ratio, dropout=dropout,
                          memory_size=memory_size, window_size=window_size)
            for _ in range(num_temporal_blocks)
        ])
        self.temporal_norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_temporal_blocks + 1)])
        
        # Temporal position embeddings: one embedding per frame up to memory_size
        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, memory_size, embed_dim))
        nn.init.trunc_normal_(self.temporal_pos_embed, std=0.02)
        
        self.head = nn.Linear(embed_dim, num_classes)
        
        self.apply(self._init_weights)
        self._init_pos_embed()
        
        self.to(device)
        print(f"InfiniViT model initialized on: {device}")
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
    
    def _init_pos_embed(self):
        if isinstance(self.img_size, int):
            h_patches = w_patches = self.img_size // self.patch_size
        else:
            h_patches = self.img_size[0] // self.patch_size
            w_patches = self.img_size[1] // self.patch_size
        
        if self.n_patches <= 1024:
            pos_embed_2d = torch.zeros(1, 1 + h_patches * w_patches, self.embed_dim)
            pos_2d = pos_embed_2d[:, 1:, :]
            position_y, position_x = torch.meshgrid(torch.arange(h_patches), torch.arange(w_patches), indexing='ij')
            pos_y = position_y.flatten() / max(h_patches - 1, 1)
            pos_x = position_x.flatten() / max(w_patches - 1, 1)
            dim = torch.arange(self.embed_dim // 2, dtype=torch.float32)
            dim = 10000 ** (2 * (dim // 2) / (self.embed_dim // 2))
            pos_y = pos_y.unsqueeze(1) / dim
            pos_x = pos_x.unsqueeze(1) / dim
            pos_y = torch.stack((torch.sin(pos_y[:, 0::2]), torch.cos(pos_y[:, 1::2])), dim=2).flatten(1)
            pos_x = torch.stack((torch.sin(pos_x[:, 0::2]), torch.cos(pos_x[:, 1::2])), dim=2).flatten(1)
            pos = torch.cat((pos_y, pos_x), dim=1)
            pos_2d.copy_(pos.unsqueeze(0))
            self.patch_embed.pos_embed.data.copy_(pos_embed_2d)
    
    def forward(self, x):
        """
        Process a sequence of images through spatial and temporal transformers.
        Expects input x of shape (T, C, H, W) where T is the number of frames.
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(np.array(x), dtype=torch.float32, device=self.device)
        x = x.to(self.device)
        
        T, C, H, W = x.shape
        target_size = self.img_size if isinstance(self.img_size, tuple) else (self.img_size, self.img_size)
        if H != target_size[0] or W != target_size[1]:
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        
        # Batch process spatially
        x = self.patch_embed(x)  # (T, n_patches+1, embed_dim)
        for i, block in enumerate(self.spatial_blocks):
            x_norm = self.spatial_norms[i](x)
            x = x + block(x_norm)
        x = self.spatial_norms[-1](x)
        
        # Flatten tokens per frame to form a single sequence for temporal processing
        T, N, D = x.shape
        x = x.reshape(1, T * N, D)  # (1, T*N, embed_dim)
        
        # Apply temporal position embeddings (vectorized expansion)
        temp_pos = self.temporal_pos_embed[:, :T, :]  # (1, T, embed_dim)
        temp_pos = temp_pos.unsqueeze(2).expand(1, T, N, D).reshape(1, T * N, D)
        x = x + temp_pos
        
        # Process with temporal transformer blocks
        for i, block in enumerate(self.temporal_blocks):
            x_norm = self.temporal_norms[i](x)
            x = x + block(x_norm)
        
        # Global average pooling over all tokens for final feature representation
        x = x.mean(dim=1)  # (1, embed_dim)
        x = x.squeeze(0)
        logits = self.head(x)
        return logits
    
    def act(self, obs):
        """
        Given an observation, sample an action.
        Expects input of shape (T, C, H, W) where T is the number of frames.
        """
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(np.array(obs), dtype=torch.float32, device=self.device)
        obs = obs.to(self.device)
            
        if obs.max() > 1.0:
            obs = obs / 255.0
            
        if obs.dim() > 4:
            obs = obs.squeeze(0)
    
        logits = self.forward(obs)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.cpu().item(), log_prob.cpu()
    
    @staticmethod
    def get_device(device=None):
        if device is not None:
            return torch.device(device)
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Usage example
if __name__ == "__main__":
    img_size = 224
    patch_size = 16
    temporal_window = 4  # Example temporal window size
    device = InfiniViT.get_device()
    print(f"Using device: {device}")
    
    # Create a random tensor of shape (T, C, H, W)
    x = torch.randn(temporal_window, 3, img_size, img_size, device=device)
    
    model = InfiniViT(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=3,
        num_classes=1000,
        embed_dim=768,
        num_heads=12,
        dropout=0.1,
        device=device,
        memory_size=64,
        window_size=8,
        num_spatial_blocks=3,
        num_temporal_blocks=3,
    )
    
    output = model(x)
    print(f"Output shape: {output.shape}")  # Should be (num_classes)