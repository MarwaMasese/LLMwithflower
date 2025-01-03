import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import flwr as fl
import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    vocab_size: int = 50257  
    hidden_size: int = 256   # Reduced size for testing
    num_layers: int = 2      # Reduced layers for testing
    num_heads: int = 4       
    intermediate_size: int = 512
    max_position_embeddings: int = 512
    layer_norm_epsilon: float = 1e-5
    dropout_prob: float = 0.1

class SimpleTransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_heads,
            dim_feedforward=config.intermediate_size,
            dropout=config.dropout_prob
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, config.num_layers)
        self.output = nn.Linear(config.hidden_size, config.vocab_size)
        
    def forward(self, x):
        x = self.embeddings(x)
        x = x.transpose(0, 1)  # TransformerEncoder expects seq_len first
        x = self.transformer(x)
        x = x.transpose(0, 1)  # Batch first again
        return self.output(x)

class SimpleTextDataset(Dataset):
    def __init__(self, size=1000, seq_length=32):
        self.size = size
        self.seq_length = seq_length
        # Generate random data
        self.data = torch.randint(0, 50257, (size, seq_length))
        self.labels = torch.randint(0, 50257, (size, seq_length))
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, test_loader):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters, config):
        print("Starting training round")
        self.set_parameters(parameters)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        self.model.train()
        total_loss = 0.0
        batches = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            if batch_idx >= 10:  # Limit batches for testing
                break
                
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            output = self.model(data)
            loss = F.cross_entropy(output.view(-1, output.size(-1)), target.view(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batches += 1
            print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
            
        print(f"Training round complete. Average loss: {total_loss/batches:.4f}")
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}
    
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        loss = 0.0
        batches = 0
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss += F.cross_entropy(output.view(-1, output.size(-1)), target.view(-1)).item()
                batches += 1
                
        average_loss = loss / batches
        print(f"Evaluation complete. Average loss: {average_loss:.4f}")
        return loss, len(self.test_loader.dataset), {"loss": average_loss}

def main():
    print("Initializing model and datasets...")
    
    # Initialize model
    config = ModelConfig()
    model = SimpleTransformer(config)
    
    # Create datasets
    train_dataset = SimpleTextDataset(size=100, seq_length=32)
    test_dataset = SimpleTextDataset(size=20, seq_length=32)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8)
    
    # Initialize and start client
    client = FlowerClient(model, train_loader, test_loader)
    print("Starting Flower client...")
    
    # Update the server address to use localhost
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",  # Changed from 0.0.0.0 to 127.0.0.1
        client=client
    )

if __name__ == "__main__":
    main()