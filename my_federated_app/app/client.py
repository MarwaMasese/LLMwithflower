"""Client module with DDP training."""
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from flwr.client import ClientApp, NumPyClient
from .task import load_model, load_tokenizer, get_data

class DDPClient(NumPyClient):
    def __init__(self, rank: int, world_size: int):
        # DDP setup
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
        
        # Model setup
        self.model = DDP(load_model(), device_ids=[rank])
        self.tokenizer = load_tokenizer()
        self.data = get_data()

    def get_parameters(self, config):
        if dist.get_rank() == 0:
            return [p.cpu().numpy() for p in self.model.parameters()]
        return []

    def fit(self, parameters, config):
        # Sync parameters
        if parameters:
            param_tensors = [torch.tensor(p).to(dist.get_rank()) for p in parameters]
            with torch.no_grad():
                for param, new_param in zip(self.model.parameters(), param_tensors):
                    param.copy_(new_param)
        
        # Training
        inputs = self.tokenizer(
            self.data, 
            return_tensors="pt", 
            padding=True,
            truncation=True,
            max_length=128
        ).to(dist.get_rank())

        outputs = self.model(**inputs, labels=inputs["input_ids"])
        outputs.loss.backward()
        
        # Only rank 0 returns updates
        return self.get_parameters(config), len(self.data), {"perplexity": outputs.loss.item()}

def client_fn(cid: str):
    world_size = 2  # GPUs per client
    rank = int(cid) % world_size
    return DDPClient(rank, world_size).to_client()

app = ClientApp(client_fn=client_fn)