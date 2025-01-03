import math
import os
import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    vocab_size: int = 50257  # GPT-2 vocabulary size
    hidden_size: int = 2048   # Hidden dimension
    num_layers: int = 24      # Number of transformer layers
    num_heads: int = 16       # Number of attention heads
    intermediate_size: int = 8192  # FFN intermediate size
    max_position_embeddings: int = 2048
    layer_norm_epsilon: float = 1e-5
    dropout_prob: float = 0.1

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])

    def forward(self, x, seq_len=None):
        return (
            self.cos_cached[:, :, :seq_len, ...],
            self.sin_cached[:, :, :seq_len, ...]
        )

def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class AttentionLayer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_heads
        
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size)
        
        self.rotary_emb = RotaryEmbedding(self.head_dim, config.max_position_embeddings)
        self.dropout = nn.Dropout(config.dropout_prob)
        
    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_length = hidden_states.shape[:2]
        
        q = self.q_proj(hidden_states).view(batch_size, seq_length, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(batch_size, seq_length, self.num_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(batch_size, seq_length, self.num_heads, self.head_dim)
        
        q = q.transpose(1, 2)  # (batch, num_heads, seq_len, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Apply rotary embeddings
        cos, sin = self.rotary_emb(hidden_states, seq_length)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Scaled dot-product attention
        scaling_factor = self.head_dim ** -0.5
        attention_scores = torch.matmul(q * scaling_factor, k.transpose(-2, -1))
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
            
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        output = torch.matmul(attention_probs, v)
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_length, self.hidden_size)
        
        return self.o_proj(output)

class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention = AttentionLayer(config)
        self.ln1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.ln2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(config.dropout_prob)
        )
        
    def forward(self, x, attention_mask=None):
        x = x + self.attention(self.ln1(x), attention_mask)
        x = x + self.mlp(self.ln2(x))
        return x

class LLM_158B(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout_prob)
        
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])
        
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
    def forward(self, input_ids, attention_mask=None):
        x = self.token_embeddings(input_ids)
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x, attention_mask)
            
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        return logits

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, test_loader, device, config):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.config = config
        
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=1e-4,
            betas=(0.9, 0.95),
            weight_decay=0.1
        )
        
        self.model.train()
        for epoch in range(1):  # Local epochs
            for batch_idx, (input_ids, labels) in enumerate(self.train_loader):
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(input_ids)
                loss = F.cross_entropy(outputs.view(-1, self.config.vocab_size), labels.view(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
        self.get_parameters(config={}), len(self.train_loader.dataset), {}

def init_distributed(rank, world_size):
    """
    Initialize the distributed process group with gloo backend
    """
    # Using gloo backend instead of nccl for better compatibility
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize process group
    dist.init_process_group(
        backend='gloo',  # Changed from nccl to gloo
        init_method='env://',  # Using environment variables instead of TCP
        world_size=world_size,
        rank=rank
    )
    
    # Set device
    torch.cuda.set_device(rank)

def main(rank, world_size):
    # Initialize distributed training
    init_distributed(rank, world_size)
    
    # Get device
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    
    # Initialize model config and model
    config = ModelConfig()
    model = LLM_158B(config)
    model = model.to(device)
    
    # Wrap model with DDP
    model = DDP(model, device_ids=[rank])
    
    # Create dummy dataset and dataloaders
    # Replace this with your actual dataset
    train_loader = DataLoader(...)  # Your training dataset
    test_loader = DataLoader(...)   # Your test dataset
    
    # Initialize Flower client
    client = FlowerClient(model, train_loader, test_loader, device, config)
    
    # Start Flower client
    fl.client.start_numpy_client(
    server_address="0.0.0.0:8080",  # Updated address
    client=client
    )

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size,), nprocs=world_size)