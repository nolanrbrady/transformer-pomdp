import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PatchEmbedding(nn.Module):
    """
    Image to Patch Embedding with position embeddings.
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768, device=None, pad_if_needed=True):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.pad_if_needed = pad_if_needed
        self.device = device
        
        # Calculate target dimensions that are divisible by patch_size
        if isinstance(img_size, int):
            self.target_height = self.target_width = img_size
            self.h_patches = self.w_patches = img_size // patch_size
        else:
            self.target_height, self.target_width = img_size
            self.h_patches = self.target_height // patch_size
            self.w_patches = self.target_width // patch_size
            
        # Calculate number of patches after potential padding
        self.n_patches = self.h_patches * self.w_patches
        
        # Create projection (equivalent to rearranging + linear layer)
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        
        # Position embeddings for patches + class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches + 1, embed_dim))
        
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
    
    def forward(self, x):
        """Process a single image without batch dimension
        Input shape: (C, H, W) - Single image without batch dimension
        Output shape: (1, n_patches+1, embed_dim) - Single sequence with class token
        """
        
        # Extract dimensions
        C, H, W = x.shape
        
        # Add padding if needed to make dimensions divisible by patch_size
        pad_h = pad_w = 0
        if self.pad_if_needed:
            pad_h = (self.patch_size - H % self.patch_size) % self.patch_size
            pad_w = (self.patch_size - W % self.patch_size) % self.patch_size
            
            if pad_h > 0 or pad_w > 0:
                x = F.pad(x, (0, pad_w, 0, pad_h))
                print(f"Padded input from {(H, W)} to {(H+pad_h, W+pad_w)}")
        
        # Add batch dimension for convolution
        x = x.unsqueeze(0)  # (1, C, H, W)
        
        # Create patches and project
        x = self.proj(x)                              # (1, embed_dim, grid_h, grid_w)
        grid_h, grid_w = x.shape[-2], x.shape[-1]     # Get actual grid dimensions
        x = x.flatten(2)                              # (1, embed_dim, n_patches)
        x = x.transpose(1, 2)                         # (1, n_patches, embed_dim)
        
        # Add class token
        cls_tokens = self.cls_token                   # (1, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)         # (1, n_patches+1, embed_dim)
        
        # Add position embedding
        # If the number of patches has changed due to padding, interpolate the position embeddings
        if x.size(1) != self.pos_embed.size(1):
            cls_pos_embed = self.pos_embed[:, 0:1, :]
            patch_pos_embed = self.pos_embed[:, 1:, :]
            
            # Calculate actual grid size after padding
            n_h = (H + pad_h) // self.patch_size
            n_w = (W + pad_w) // self.patch_size
            
            # Reshape using the original grid dimensions stored in the class
            patch_pos_embed = patch_pos_embed.reshape(1, self.h_patches, self.w_patches, -1)
            patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
            
            # Interpolate to the new grid size
            patch_pos_embed = F.interpolate(patch_pos_embed, size=(n_h, n_w), mode='bilinear')
            patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).flatten(1, 2)
            
            # Combine with class token position embedding
            pos_embed = torch.cat([cls_pos_embed, patch_pos_embed], dim=1)
            x = x + pos_embed
        else:
            x = x + self.pos_embed                    # (1, n_patches+1, embed_dim)
        
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

class Block(nn.Module):
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
    
class InfiniAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, memory_size=64, window_size=8, dropout=0.1, ema_decay=0.9):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.memory_size = memory_size
        self.window_size = window_size
        self.head_dim = embed_dim // num_heads
        self.ema_decay = ema_decay  # EMA decay factor (between 0 and 1)

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        # For memory: learned compressive projection
        self.memory_proj = nn.Linear(embed_dim, embed_dim)
        self.register_buffer("memory", torch.zeros(1, memory_size, embed_dim))  # (1, M, D)
        self.register_buffer("memory_initialized", torch.tensor(False))  # Flag for first-time init

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv  # (B, heads, T, head_dim)

        # Local attention (within the current window)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_scores, dim=-1)
        local_out = torch.matmul(attn_weights, v)  # (B, heads, T, head_dim)

        # Memory attention
        mem = self.memory_proj(self.memory)  # (1, M, D)
        mem = mem.expand(B, -1, -1)  # Broadcast to batch
        mem_kv = mem.reshape(B, self.memory_size, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        mem_attn = torch.matmul(q, mem_kv.transpose(-2, -1)) / np.sqrt(self.head_dim)
        mem_attn_weights = F.softmax(mem_attn, dim=-1)
        mem_out = torch.matmul(mem_attn_weights, mem_kv)

        # Combine local and memory attention outputs
        out = local_out + mem_out
        out = out.transpose(1, 2).reshape(B, T, C)
        out = self.out_proj(out)

        # === EMA Memory Update ===
        # Create a new memory slot from most recent tokens
        new_summary = x[:, -self.window_size:, :].mean(dim=1, keepdim=True)  # (B, 1, D)

        # If memory hasn't been initialized, just fill it with the first summary
        if not self.memory_initialized:
            self.memory = new_summary.mean(dim=0, keepdim=True).repeat(1, self.memory_size, 1).detach()
            self.memory_initialized = torch.tensor(True, device=x.device)
        else:
            # EMA update: slide every slot one step, and blend the newest into the last slot
            ema_updated = self.ema_decay * self.memory[:, -1:, :] + (1 - self.ema_decay) * new_summary.mean(dim=0, keepdim=True)
            self.memory = torch.cat([
                self.memory[:, 1:],   # Drop oldest
                ema_updated.detach()  # Add newest EMA slot
            ], dim=1)

        return out
    
class TemporalBlock(nn.Module):
    """
    Temporal Transformer Block: Multi-Head Attention + MLP with LayerNorm.
    """
    def __init__(self, embed_dim, num_heads, memory_size=64, window_size=8, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = InfiniAttention(embed_dim, num_heads, memory_size=memory_size, window_size=window_size, dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.memory_size = memory_size
        self.window_size = window_size
        self.mlp = MLP(
            in_features=embed_dim,
            hidden_features=int(embed_dim * mlp_ratio),
            out_features=embed_dim,
            dropout=dropout,
        )
    
    def forward(self, x):
        """
        Forward pass for the TemporalBlock.
        X: (B, C, H, W)
        """
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
        

class InfiniViT(nn.Module):
    """
    Infini Vision Transformer with a single transformer block.
    
    Args:
        img_size (int or tuple): Input image size (height, width).
        patch_size (int): Patch size.
        in_channels (int): Number of input channels.
        num_classes (int): Number of classes for classification.
        embed_dim (int): Embedding dimension. 
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        dropout (float): Dropout rate.
        pad_if_needed (bool): Whether to pad input images to ensure dimensions are divisible by patch_size.
        device (str or torch.device, optional): Device to use ('cpu', 'cuda'). If None, uses cuda if available.
    """
    def __init__(
        self,
        img_size=224,
        patch_size=16,
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
    ):
        super().__init__()
        
        # Set device
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        
        # Save parameters
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Attention parameters
        self.memory_size = memory_size
        self.window_size = window_size
        
        # Handle both int and tuple image sizes
        if isinstance(img_size, int):
            h_patches = w_patches = img_size // patch_size
        else:
            h_patches = img_size[0] // patch_size
            w_patches = img_size[1] // patch_size
            
        # Calculate number of patches
        self.n_patches = h_patches * w_patches
        print(f"Using grid of {h_patches}x{w_patches} patches = {self.n_patches} total patches")
        
        # Patch + Position Embedding
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            pad_if_needed=pad_if_needed,
            device=device,
        )
        
        # Spatial transformer block for individual frames
        self.spatial_block = Block(
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
        
        # Temporal transformer block for sequence of frames
        self.temporal_block = TemporalBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            memory_size=memory_size,
            window_size=window_size,
        )
        
        # Temporal position embeddings
        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, memory_size, embed_dim))  # Support up to 100 frames
        nn.init.trunc_normal_(self.temporal_pos_embed, std=0.02)
        
        # Layer Norm
        self.spatial_norm = nn.LayerNorm(embed_dim)
        self.temporal_norm = nn.LayerNorm(embed_dim)
        
        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Weight initialization
        self.apply(self._init_weights)
        
        # Initialize position embeddings with proper 2D structure
        self._init_pos_embed()
        
        # Move model to device
        self.to(device)
        print(f"TemporalViT model initialized on: {device}")
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
    
    def _init_pos_embed(self):
        """Initialize the position embeddings with a more structured 2D pattern."""
        # Handle both int and tuple image sizes
        if isinstance(self.img_size, int):
            h_patches = w_patches = self.img_size // self.patch_size
        else:
            h_patches = self.img_size[0] // self.patch_size
            w_patches = self.img_size[1] // self.patch_size
        
        # Get the position embedding from patch_embed
        pos_embed = self.patch_embed.pos_embed
        
        # Only initialize if we're using a reasonable number of patches
        if self.n_patches <= 1024:  # Arbitrary limit to prevent huge position embeddings
            # Create 2D positional encodings
            pos_embed_2d = torch.zeros(1, 1 + h_patches * w_patches, self.embed_dim)
            
            # Skip the class token
            pos_2d = pos_embed_2d[:, 1:, :]
            
            # Initialize with simple 2D sine/cosine positional encoding
            position_y, position_x = torch.meshgrid(
                torch.arange(h_patches), 
                torch.arange(w_patches),
                indexing='ij'
            )
            
            # Flatten and normalize position coordinates to [0, 1]
            pos_y = position_y.flatten() / max(h_patches - 1, 1)
            pos_x = position_x.flatten() / max(w_patches - 1, 1)
            
            # Scale by embedding dimension
            dim = torch.arange(self.embed_dim // 2, dtype=torch.float32)
            dim = 10000 ** (2 * (dim // 2) / (self.embed_dim // 2))
            
            # Create the positional encodings
            pos_y = pos_y.unsqueeze(1) / dim
            pos_x = pos_x.unsqueeze(1) / dim
            
            pos_y = torch.stack((torch.sin(pos_y[:, 0::2]), torch.cos(pos_y[:, 1::2])), dim=2).flatten(1)
            pos_x = torch.stack((torch.sin(pos_x[:, 0::2]), torch.cos(pos_x[:, 1::2])), dim=2).flatten(1)
            
            # Combine x and y encodings
            pos = torch.cat((pos_y, pos_x), dim=1)
            pos_2d.copy_(pos.unsqueeze(0))
            
            # Copy back to the patch_embed's position embeddings
            self.patch_embed.pos_embed.data.copy_(pos_embed_2d)
    
    def process_single_image(self, img):
        """
        Process a single image through patch embedding and spatial transformer.
        
        Args:
            img: Single image of shape (C, H, W) or (1, C, H, W)
            
        Returns:
            Embedded representation of shape (1, n_patches+1, embed_dim)
        """
        # Ensure input is on the correct device
        if img.device != self.device:
            img = img.to(self.device)
            
        # Handle singleton dimension if present
        if len(img.shape) == 4 and img.shape[0] == 1:
            img = img.squeeze(0)
        
        # Patch embedding
        x = self.patch_embed(img)  # (1, n_patches+1, embed_dim)
        
        # Apply spatial transformer block
        x = self.spatial_block(x)  # (1, n_patches+1, embed_dim)
        
        # Apply layer norm
        x = self.spatial_norm(x)  # (1, n_patches+1, embed_dim)
        
        return x
    
    def forward(self, x):
        """
        Process a sequence of images through spatial and temporal transformers.
        
        Args:
            x: Temporal sequence of frames with shape (T, C, H, W) where:
               - T is the temporal window size (e.g., 4 recent frames)
               - C is the number of channels
               - H, W are the height and width of each frame
            
        Returns:
            Class logits of shape (num_classes)
            
        Process:
            1. Each frame is processed individually through spatial embedding and transformer
            2. CLS tokens from all frames are collected to form a temporal sequence
            3. Temporal sequence is processed through a temporal transformer
            4. Final prediction is made from the last temporal token
        """
        # Ensure input is a tensor on the correct device
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(np.array(x), dtype=torch.float32, device=self.device)
        elif x.device != self.device:
            x = x.to(self.device)
        
        # Process each image through spatial embedding and transformer
        T = x.shape[0]  # Number of frames in sequence
        
        # Process images one by one (batch-size agnostic)
        spatial_embeddings = []
        for t in range(T):
            img = x[t]  # (C, H, W)
            embedded_img = self.process_single_image(img)  # (1, n_patches+1, embed_dim)
            
            # Extract CLS token representation
            cls_token = embedded_img[:, 0:1, :]  # (1, 1, embed_dim)
            spatial_embeddings.append(cls_token)
        
        # Stack all CLS token embeddings along temporal dimension
        temporal_sequence = torch.cat(spatial_embeddings, dim=1)  # (1, T, embed_dim)
        
        # Add temporal position embeddings
        temporal_pos_embed = self.temporal_pos_embed[:, :T, :]
        temporal_sequence = temporal_sequence + temporal_pos_embed
        
        # Apply temporal transformer block
        temporal_sequence = self.temporal_block(temporal_sequence)  # (1, T, embed_dim)
        
        # Apply layer norm
        temporal_sequence = self.temporal_norm(temporal_sequence)  # (1, T, embed_dim)
        
        # Use the last token for classification
        x = self.head(temporal_sequence[:, -1])  # (1, num_classes)
        
        # Remove batch dimension
        x = x.squeeze(0)  # (num_classes)
        
        return x
    
    def act(self, obs):
        """
        Given an observation, sample an action.
        Expects input of shape (T, C, H, W) where T is the number of frames.
        
        Returns the action and the log probability of that action.
        """ 
        # Make sure input is a tensor on the correct device
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(np.array(obs), dtype=torch.float32, device=self.device)
        elif obs.device != self.device:
            obs = obs.to(self.device)
            
        # Normalize if not already done
        if obs.max() > 1.0:
            obs = obs / 255.0
            
        # Remove any extra dimensions if needed
        if len(obs.shape) > 4:  # If shape is (B, T, C, H, W)
            obs = obs.squeeze(0)  # Remove batch dimension if present
    
        # Forward pass
        logits = self.forward(obs)  # (num_classes)
        
        # Compute action probabilities
        probs = torch.softmax(logits, dim=-1)  # (num_classes)
        
        # Create categorical distribution
        dist = torch.distributions.Categorical(probs)
        
        # Sample action
        action = dist.sample()  # scalar
        
        # Get log probability of the action
        log_prob = dist.log_prob(action)  # scalar
        
        # Move action and log_prob to CPU for returning
        action_cpu = action.cpu()
        log_prob_cpu = log_prob.cpu()
        
        # Return action as int and log_prob as tensor (for backpropagation)
        return action_cpu.item(), log_prob_cpu
        
    @staticmethod
    def get_device(device=None):
        """
        Helper method to get the best available device.
        
        Args:
            device: Specified device or None to auto-detect
            
        Returns:
            torch.device: The device to use
        """
        if device is not None:
            return torch.device(device)
        
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')

# Usage example (for reference only)
if __name__ == "__main__":
    # Example usage
    img_size = 224
    patch_size = 16
    batch_size = 4
    
    # Get device
    device = TemporalViT.get_device()
    print(f"Using device: {device}")
    
    # Create a random tensor of size (batch_size, channels, height, width)
    x = torch.randn(batch_size, 3, img_size, img_size, device=device)
    
    # Initialize the model
    model = TemporalViT(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=3,
        num_classes=1000,
        embed_dim=768,
        num_heads=12,
        dropout=0.1,
        device=device,
    )
    
    # Forward pass
    output = model(x)
    print(f"Output shape: {output.shape}")  # Should be (batch_size, num_classes)