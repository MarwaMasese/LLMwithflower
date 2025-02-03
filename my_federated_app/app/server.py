"""Server module for federated LLM training."""
from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.server import ServerApp, ServerConfig
from flwr.server.strategy import FedAvg
from .task import load_model

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate client perplexity metrics."""
    perplexities = [num_examples * m["perplexity"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"perplexity": sum(perplexities) / sum(examples)}

def server_fn(context: Context):
    """Define server behavior."""
    # Load initial model parameters
    model = load_model()
    initial_params = [param.detach().cpu().numpy() for param in model.parameters()]
    
    strategy = FedAvg(
        fraction_fit=1.0,
        min_available_clients=2,
        initial_parameters=ndarrays_to_parameters(initial_params),
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    return ServerAppComponents(
        strategy=strategy,
        config=ServerConfig(num_rounds=3)
    )

app = ServerApp(server_fn=server_fn)