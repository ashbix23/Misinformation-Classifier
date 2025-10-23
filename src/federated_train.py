import os, flwr as fl, torch
from src.config import Paths, DATASETS
from src.fl_client import FakeNewsClient

def client_fn(cid: str) -> fl.client.Client:
    idx = int(cid)  # 0..8
    datasets = ["ISOT","FakeNews","LIAR"]
    ds = datasets[idx // 3]
    csv = os.path.join(Paths().processed_dir, DATASETS[ds])
    return FakeNewsClient(csv).to_client()

if __name__ == "__main__":
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.5, fraction_evaluate=1.0,
        min_fit_clients=3, min_evaluate_clients=3, min_available_clients=9,
    )
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=9,
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
        ray_init_args={"include_dashboard": False},
    )

