# app/client.py
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM, AutoTokenizer
import flwr as fl

class FlowerClient(fl.client.NumPyClient):
    def __init__(self):
        # Initialize DDP and model (adapt your earlier code)
        self.rank = 0  # Example - replace with your DDP setup
        self.model = AutoModelForCausalLM.from_pretrained(
            "EleutherAI/gpt-neo-1.3B",
            device_map="auto",
            load_in_4bit=True
        )
        self.model = DDP(self.model, device_ids=[self.rank])

    def get_parameters(self, config):
        # Return parameters from rank 0
        if self.rank == 0:
            return [p.cpu().numpy() for p in self.model.parameters()]
        return []

    def fit(self, parameters, config):
        # Your training logic here
        return self.get_parameters(config), 100, {}

def client_fn(cid: str) -> FlowerClient:
    return FlowerClient()

# New Flower CLI requires this function
def start_client():
    fl.client.start_client(server_address="127.0.0.1:8080", client=FlowerClient())