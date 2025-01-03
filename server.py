import flwr as fl
from typing import List, Tuple, Dict, Optional
from flwr.common import Metrics, FitIns, Parameters
from flwr.server.client_proxy import ClientProxy

class CustomStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitIns]],
        failures: List[BaseException],
    ) -> Optional[Parameters]:
        print(f"Round {server_round} completed")
        return super().aggregate_fit(server_round, results, failures)

# Start server with localhost address
strategy = CustomStrategy(
    min_fit_clients=1,
    min_available_clients=1,
)

print("Starting Flower server...")
fl.server.start_server(
    server_address="127.0.0.1:8080",  # Changed from 0.0.0.0 to 127.0.0.1
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=strategy,
)