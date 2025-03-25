import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTConfig, ViTForImageClassification
import torchvision.transforms as transforms

class TinyViTPolicy(nn.Module):
    """
    A lightweight Vision Transformer (ViT) implementation for reinforcement learning.
    Uses a smaller, non-pretrained ViT model to reduce computational requirements.
    """
    def __init__(
        self,
        img_size=(224, 224),
        in_channels=3,
        num_classes=3,
        resize_inputs=True,
        config_override=None
    ):
        super().__init__()
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.resize_inputs = resize_inputs
        
        # Define a tiny ViT configuration if none provided
        if config_override is None:
            config = ViTConfig(
                image_size=224,  # ViT standard size
                patch_size=16,   # Standard patch size
                num_channels=in_channels,
                num_labels=num_classes,
                hidden_size=192,  # Much smaller than base (768)
                num_hidden_layers=6,  # Base has 12
                num_attention_heads=3,  # Base has 12
                intermediate_size=768,  # Base has 3072 
                hidden_dropout_prob=0.1,
                attention_probs_dropout_prob=0.1,
            )
        else:
            config = config_override
            
        # Create a non-pretrained model
        self.vit_model = ViTForImageClassification(config)
        
        # Standard ImageNet normalization
        self.image_mean = [0.485, 0.456, 0.406] 
        self.image_std = [0.229, 0.224, 0.225]
        
        # For grayscale images, we need to repeat the channels to get RGB
        self.needs_channel_repeat = (in_channels == 1)
        
        # For resizing input images to 224x224 (ViT's expected size)
        if resize_inputs:
            self.resize_transform = transforms.Resize(
                (224, 224), 
                interpolation=transforms.InterpolationMode.BICUBIC
            )
        
        # Normalization transform
        self.normalize = transforms.Normalize(mean=self.image_mean, std=self.image_std)
    
    def preprocess_image(self, x):
        """
        Preprocess an image tensor for the ViT model, maintaining gradient flow.
        x: Tensor of shape (C, H, W) or (B, C, H, W)
        """
        # Add batch dimension if not present
        if len(x.shape) == 3:
            x = x.unsqueeze(0)  # (1, C, H, W)
        
        # If input is grayscale, repeat to 3 channels
        if self.needs_channel_repeat:
            x = x.repeat(1, 3, 1, 1)
        
        # Resize if needed
        if self.resize_inputs:
            B, C, H, W = x.shape
            x_resized = torch.zeros(B, C, 224, 224, device=x.device, requires_grad=x.requires_grad)
            for i in range(B):
                x_resized[i] = self.resize_transform(x[i])
            x = x_resized
        
        # Ensure values are in [0, 1] range
        if x.min() < 0 or x.max() > 1.0:
            x = torch.clamp(x, 0.0, 1.0)
        
        # Apply normalization
        x = self.normalize(x)
        
        return {"pixel_values": x}
    
    def forward(self, x):
        """
        Forward pass through the ViT model.
        x: Tensor of shape (C, H, W) or (B, C, H, W)
        Returns: logits of shape (B, num_classes)
        """
        inputs = self.preprocess_image(x)
        outputs = self.vit_model(**inputs)
        return outputs.logits
    
    def act(self, x, training=False):
        """
        Sample an action based on input observation.
        For REINFORCE, returns action and log probability.
        x: Tensor of shape (C, H, W) or (B, C, H, W)
        training: If True, gradient computation is enabled
        """
        if not training:
            with torch.no_grad():
                logits = self.forward(x)
                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
            return action.item(), log_prob
        else:
            # During training, we need to keep gradients
            logits = self.forward(x)
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            return action.item(), log_prob

    def evaluate(self, x, actions):
        """
        Evaluate actions for given states, returning log probabilities and entropy.
        For PPO and Actor-Critic methods.
        x: Tensor of shape (B, C, H, W)
        actions: Tensor of shape (B,) containing action indices
        """
        logits = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_probs, entropy


class TinyViTActorCriticPolicy(TinyViTPolicy):
    """
    Extends the TinyViTPolicy to include a value head for Actor-Critic methods.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Add a value head for critic
        embed_dim = self.vit_model.config.hidden_size
        self.value_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # Initialize the value head weights
        for m in self.value_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Forward pass through both the actor (policy) and critic (value) networks.
        x: Tensor of shape (C, H, W) or (B, C, H, W)
        Returns: 
            logits: action logits of shape (B, num_classes)
            values: state value estimates of shape (B, 1)
        """
        inputs = self.preprocess_image(x)
        outputs = self.vit_model(**inputs, output_hidden_states=True)
        logits = outputs.logits
        
        # Get the [CLS] token embedding for the value head
        hidden_states = outputs.hidden_states[-1][:, 0]  # Use CLS token from last layer
        values = self.value_head(hidden_states)
        
        return logits, values.squeeze(-1)
    
    def act(self, x, training=False):
        """
        Sample an action and return log probability and value estimate.
        For Actor-Critic methods.
        x: Tensor of shape (C, H, W) or (B, C, H, W)
        training: If True, gradient computation is enabled
        """
        if not training:
            with torch.no_grad():
                logits, value = self.forward(x)
                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
            return action.item(), log_prob, value
        else:
            # During training, we need to keep gradients
            logits, value = self.forward(x)
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            return action.item(), log_prob, value
    
    def evaluate(self, x, actions):
        """
        Evaluate actions for given states, returning log probabilities, values, and entropy.
        For PPO and Actor-Critic methods.
        x: Tensor of shape (B, C, H, W)
        actions: Tensor of shape (B,) containing action indices
        """
        logits, values = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_probs, values, entropy


# Add backward compatibility aliases
HuggingFaceViTPolicy = TinyViTPolicy
HuggingFaceActorCriticPolicy = TinyViTActorCriticPolicy


# Example usage
if __name__ == "__main__":
    # Create a simple test image
    x = torch.randn(3, 84, 84)  # Single image, 3 channels, 84x84 (common in RL)
    
    # Test Policy (actor only)
    policy = TinyViTPolicy(
        img_size=(84, 84),
        in_channels=3,
        num_classes=4,  # 4 actions
        resize_inputs=True  # Resize to 224x224 for ViT
    )
    
    # Test forward
    with torch.no_grad():
        logits = policy(x)
        print(f"Logits shape: {logits.shape}")
    
    # Test act
    action, log_prob = policy.act(x)
    print(f"Action: {action}, Log prob: {log_prob.item()}")
    
    # Test Actor-Critic Policy
    ac_policy = TinyViTActorCriticPolicy(
        img_size=(84, 84),
        in_channels=3,
        num_classes=4,
        resize_inputs=True
    )
    
    # Test forward
    with torch.no_grad():
        logits, value = ac_policy(x)
        print(f"Logits shape: {logits.shape}, Value: {value.item()}")
    
    # Test act
    action, log_prob, value = ac_policy.act(x)
    print(f"Action: {action}, Log prob: {log_prob.item()}, Value: {value.item()}") 