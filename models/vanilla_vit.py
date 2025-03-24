import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    """
    Image to Patch Embedding with position embeddings.
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768, pad_if_needed=True):
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
        

class SimpleViT(nn.Module):
    """
    Simple Vision Transformer with a single transformer block.
    
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
    ):
        super().__init__()
        
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
        )
        
        # Only one transformer block, as requested
        self.block = Block(
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
        
        # Layer Norm
        self.norm = nn.LayerNorm(embed_dim)
        
        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Weight initialization
        self.apply(self._init_weights)
        
        # Initialize position embeddings with proper 2D structure
        self._init_pos_embed()
    
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
    
    def forward(self, x):
        """
        Forward pass for the SimpleViT model.
        For single-image processing:
        x: (C, H, W) - Single image without batch dimension
        
        Returns: (num_classes) - Class logits
        """
        print("Input shape in SimpleViT forward:", x.shape)
        
        # Handle single image vs batch
        if len(x.shape) == 3:  # Single image: (C, H, W)
            single_image = True
            # Process single image
            x = self.patch_embed(x)  # (1, n_patches+1, embed_dim)
            
            # Apply transformer block
            x = self.block(x)  # (1, n_patches+1, embed_dim)
            
            # Apply layer norm
            x = self.norm(x)  # (1, n_patches+1, embed_dim)
            
            # Classification from CLS token
            x = self.head(x[:, 0])  # (1, num_classes)
            
            # Remove batch dimension for consistency
            x = x.squeeze(0)  # (num_classes)
            
        else:  # Batch of images: (B, C, H, W)
            single_image = False
            batch_size = x.shape[0]
            
            # Process batch
            embeddings = []
            
            for i in range(batch_size):
                # Extract single image
                img = x[i]  # (C, H, W)
                
                # Process through patch embedding
                embed = self.patch_embed(img)  # (1, n_patches+1, embed_dim)
                embeddings.append(embed)
            
            # Stack along batch dimension
            x = torch.cat(embeddings, dim=0)  # (B, n_patches+1, embed_dim)
            
            # Apply transformer block
            x = self.block(x)  # (B, n_patches+1, embed_dim)
            
            # Apply layer norm
            x = self.norm(x)  # (B, n_patches+1, embed_dim)
            
            # Classification from CLS token
            x = self.head(x[:, 0])  # (B, num_classes)
        
        print("Output shape:", x.shape)
        return x
    
    def act(self, obs):
        """
        Given an observation, sample an action.
        Expects input of shape (B, C, H, W) where B is typically 1 for RL.
        
        Returns the action and the log probability of that action.
        """ 
        print("obs.shape in act", obs.shape)
        
        # If batch dimension is 1, we can remove it for processing a single image
        if obs.shape[0] == 1:
            # Remove batch dimension: (1, C, H, W) -> (C, H, W)
            single_image = obs.squeeze(0)
            
            # Get logits from forward pass
            logits = self.forward(single_image)  # (num_classes)
            
            # Add batch dimension for softmax
            logits = logits.unsqueeze(0)  # (1, num_classes)
        else:
            # Process entire batch
            logits = self.forward(obs)  # (B, num_classes)
        
        # Compute action probabilities
        probs = torch.softmax(logits, dim=-1)  # (B, num_classes)
        
        # Create categorical distribution
        dist = torch.distributions.Categorical(probs)
        
        # Sample action
        action = dist.sample()  # (B,)
        
        # Get log probability of the action
        log_prob = dist.log_prob(action)  # (B,)
        
        # Return first action and log_prob if batch was given
        return action.item(), log_prob[0] if log_prob.shape[0] > 1 else log_prob

# Usage example (for reference only)
if __name__ == "__main__":
    # Example usage
    img_size = 224
    patch_size = 16
    batch_size = 4
    
    # Create a random tensor of size (batch_size, channels, height, width)
    x = torch.randn(batch_size, 3, img_size, img_size)
    
    # Initialize the model
    model = SimpleViT(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=3,
        num_classes=1000,
        embed_dim=768,
        num_heads=12,
        dropout=0.1,
    )
    
    # Forward pass
    output = model(x)
    print(f"Output shape: {output.shape}")  # Should be (batch_size, num_classes)