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
    
class TemporalBlock(nn.Module):
    """
    Temporal Transformer Block: Multi-Head Attention + MLP with LayerNorm.
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
        """
        Forward pass for the TemporalBlock.
        X: (B, C, H, W)
        """
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
        

class TemporalViT(nn.Module):
    """
    Temporal Vision Transformer with a single transformer block.
    
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
        mlp_ratio=4.0,
        dropout=0.0,
        pad_if_needed=True,
        device=None,
        num_spatial_blocks=3,
        num_temporal_blocks=3,
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

        # Define spatial blocks with their own normalization
        self.spatial_blocks = nn.ModuleList([
            Block(embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(num_spatial_blocks)
        ])
        
        # Define spatial norms - one per block plus final
        self.spatial_norms = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(num_spatial_blocks + 1)
        ])
        
        # Define temporal blocks with their own normalization
        self.temporal_blocks = nn.ModuleList([
            Block(embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(num_temporal_blocks)
        ])
        
        # Define temporal norms - one per block plus final
        self.temporal_norms = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(num_temporal_blocks + 1)
        ])
        
        # Temporal position embeddings
        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, 100, embed_dim))  # Support up to 100 frames
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
        Process a single image through multiple spatial blocks
        """
        # Ensure input is on the correct device
        if img.device != self.device:
            img = img.to(self.device)
            
        # Handle singleton dimension if present
        if len(img.shape) == 4 and img.shape[0] == 1:
            img = img.squeeze(0)
        
        # Patch embedding
        x = self.patch_embed(img)  # (1, n_patches+1, embed_dim)
        
        # Apply multiple spatial transformer blocks with their own norms
        for i, block in enumerate(self.spatial_blocks):
            # Pre-norm before each block
            x_norm = self.spatial_norms[i](x)
            x = x + block(x_norm)  # Residual connection
        
        # Final spatial normalization
        x = self.spatial_norms[-1](x)  # Use the last norm for final output
        
        return x
    
    def forward(self, x):
        """
        Process a sequence of images through spatial and temporal blocks
        """
        # Ensure input is a tensor on the correct device
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(np.array(x), dtype=torch.float32, device=self.device)
        elif x.device != self.device:
            x = x.to(self.device)
        
        # Process each image through spatial blocks
        spatial_embeddings = []
        for t in range(x.shape[0]):
            img = x[t]
            embedded_img = self.process_single_image(img)
            
            # Use the entire token sequence instead of just the class token
            spatial_embeddings.append(embedded_img)
        
        # Concatenate all tokens from all frames
        # If each frame has N tokens, this will create a sequence of T*N tokens
        temporal_sequence = torch.cat(spatial_embeddings, dim=1)
        
        # Need to adjust position embeddings accordingly
        # Create expanded temporal position embeddings for all tokens from each frame
        frame_lengths = [embed.size(1) for embed in spatial_embeddings]
        expanded_pos_embed = []
        for t, length in enumerate(frame_lengths):
            # Use the same temporal position embedding for all tokens from the same frame
            pos_embed_t = self.temporal_pos_embed[:, t:t+1, :].expand(-1, length, -1)
            expanded_pos_embed.append(pos_embed_t)
        expanded_pos_embed = torch.cat(expanded_pos_embed, dim=1)
        
        # Add position embeddings
        temporal_sequence = temporal_sequence + expanded_pos_embed
        
        # Process with temporal blocks as before
        for i, block in enumerate(self.temporal_blocks):
            x_norm = self.temporal_norms[i](temporal_sequence)
            temporal_sequence = temporal_sequence + block(x_norm)
        
        # Final layer norm
        temporal_sequence = self.temporal_norms[-1](temporal_sequence)
        
        # Global average pooling over all tokens for final representation
        x = self.head(temporal_sequence.mean(dim=1))
        
        return x.squeeze(0)
    
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