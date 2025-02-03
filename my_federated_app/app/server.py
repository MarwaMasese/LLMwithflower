# app/server.py
import flwr as fl

def start_server():
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,  # Use all clients
        fraction_evaluate=0.0,
    )
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=strategy,
    )