import argparse
import json
import os
import subprocess
import sys
import threading
import time
import socket
import logging
import warnings
from typing import Dict, List, Tuple

import flwr as fl
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from flwr.common import NDArrays, ndarrays_to_parameters, parameters_to_ndarrays
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S", force=True)
LOGGER = logging.getLogger("ser.flwr")
logging.getLogger("flwr").setLevel(logging.ERROR)
logging.getLogger("grpc").setLevel(logging.ERROR)


def log(message: str) -> None:
    LOGGER.info(message)


def get_model(input_dim: int, num_classes: int, hidden_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(p=0.1),
        nn.Linear(hidden_dim, num_classes),
    )


def get_parameters(model: nn.Module) -> NDArrays:
    return [val.detach().cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model: nn.Module, parameters: NDArrays) -> None:
    keys = list(model.state_dict().keys())
    state_dict = {k: torch.tensor(v) for k, v in zip(keys, parameters)}
    model.load_state_dict(state_dict, strict=True)


def add_dp_noise(arr: np.ndarray, epsilon: float, max_norm: float) -> np.ndarray:
    if epsilon <= 0 or max_norm <= 0:
        return arr
    delta = 1e-5
    norm = np.linalg.norm(arr)
    clipped = arr
    if norm > max_norm and norm > 0:
        clipped = arr * (max_norm / norm)
    sigma = (max_norm / epsilon) * np.sqrt(2 * np.log(1.25 / delta))
    noise = np.random.normal(loc=0.0, scale=sigma, size=arr.shape)
    return clipped + noise


def train_local(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    epochs: int,
    lr: float,
    device: str,
    batch_size: int,
) -> float:
    model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    x_t = torch.tensor(x, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    loader = DataLoader(TensorDataset(x_t, y_t), batch_size=batch_size, shuffle=True)

    for _ in range(epochs):
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(x_t.to(device)).argmax(dim=1).cpu()
        acc = (preds == y_t).float().mean().item()
    return acc


def evaluate_model(model: nn.Module, x: np.ndarray, y: np.ndarray, device: str) -> Tuple[float, float]:
    model.to(device)
    model.eval()
    x_t = torch.tensor(x, dtype=torch.float32, device=device)
    y_t = torch.tensor(y, dtype=torch.long, device=device)
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        logits = model(x_t)
        loss = criterion(logits, y_t).item()
        preds = logits.argmax(dim=1)
        acc = (preds == y_t).float().mean().item()
    return loss, acc


def stratified_iid_partition(y: np.ndarray, num_clients: int, seed: int) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    client_indices: List[List[int]] = [[] for _ in range(num_clients)]

    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        splits = np.array_split(idx, num_clients)
        for cid, part in enumerate(splits):
            client_indices[cid].extend(part.tolist())

    out: List[np.ndarray] = []
    for cid in range(num_clients):
        arr = np.array(client_indices[cid], dtype=np.int64)
        rng.shuffle(arr)
        out.append(arr)
    return out


def prepare_data(seed: int, test_size: float):
    data = np.load("embeddings.npz")
    x, y = data["X"], data["y"]
    num_classes = int(np.max(y) + 1)
    input_dim = int(x.shape[1])

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        stratify=y,
        random_state=seed,
    )

    if os.path.exists("scaler.pkl"):
        scaler = joblib.load("scaler.pkl")
    else:
        scaler = StandardScaler().fit(x_train)
        joblib.dump(scaler, "scaler.pkl")

    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    return x_train, x_test, y_train, y_test, input_dim, num_classes


class SERClient(fl.client.NumPyClient):
    def __init__(
        self,
        x_local: np.ndarray,
        y_local: np.ndarray,
        input_dim: int,
        num_classes: int,
        hidden_dim: int,
        local_epochs: int,
        lr: float,
        epsilon: float,
        max_norm: float,
        batch_size: int,
        device: str,
    ) -> None:
        self.x_local = x_local
        self.y_local = y_local
        self.local_epochs = local_epochs
        self.lr = lr
        self.epsilon = epsilon
        self.max_norm = max_norm
        self.batch_size = batch_size
        self.device = device
        self.model = get_model(input_dim, num_classes, hidden_dim)

    def get_parameters(self, config: Dict[str, str]) -> NDArrays:
        return get_parameters(self.model)

    def fit(self, parameters: NDArrays, config: Dict[str, str]):
        set_parameters(self.model, parameters)
        local_acc = train_local(
            self.model,
            self.x_local,
            self.y_local,
            epochs=self.local_epochs,
            lr=self.lr,
            device=self.device,
            batch_size=self.batch_size,
        )
        updated = get_parameters(self.model)
        privatized = [add_dp_noise(p, self.epsilon, self.max_norm) for p in updated]
        return privatized, len(self.y_local), {"local_acc": local_acc}

    def evaluate(self, parameters: NDArrays, config: Dict[str, str]):
        set_parameters(self.model, parameters)
        loss, acc = evaluate_model(self.model, self.x_local, self.y_local, self.device)
        return float(loss), len(self.y_local), {"local_acc": float(acc)}


class SaveFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, checkpoint_dir: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def aggregate_fit(self, server_round, results, failures):
        aggregated, metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated is not None:
            nds = parameters_to_ndarrays(aggregated)
            checkpoint = {"parameters": nds, "round": server_round}
            joblib.dump(checkpoint, os.path.join(self.checkpoint_dir, f"global_round_{server_round}.pkl"))
        return aggregated, metrics


def run_server(args) -> None:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log("=== Flower SER training start ===")
    log(f"Device: {device}")
    log(
        "Config: clients=%s rounds=%s local_epochs=%s lr=%s batch_size=%s hidden_dim=%s test_size=%s epsilon=%s max_norm=%s"
        % (
            args.num_clients,
            args.rounds,
            args.local_epochs,
            args.lr,
            args.batch_size,
            args.hidden_dim,
            args.test_size,
            args.epsilon,
            args.max_norm,
        )
    )

    x_train, x_test, y_train, y_test, input_dim, num_classes = prepare_data(args.seed, args.test_size)
    partitions = stratified_iid_partition(y_train, args.num_clients, args.seed)
    log(f"Data split: {len(x_train)} train / {len(x_test)} test")
    for cid, partition in enumerate(partitions):
        log(f"Client {cid}: {len(partition)} local samples")

    central_model = get_model(input_dim, num_classes, args.hidden_dim)
    metrics_store: Dict[str, List[Tuple[int, float]]] = {"test_acc": [], "test_loss": []}

    def evaluate_fn(server_round: int, parameters: NDArrays, config: Dict[str, str]):
        set_parameters(central_model, parameters)
        loss, acc = evaluate_model(central_model, x_test, y_test, device)
        metrics_store["test_acc"].append((server_round, float(acc)))
        metrics_store["test_loss"].append((server_round, float(loss)))
        log(f"Round {server_round}: global test_acc={acc:.4f} loss={loss:.4f}")
        return float(loss), {"test_acc": float(acc)}

    initial_parameters = ndarrays_to_parameters(get_parameters(central_model))

    strategy = SaveFedAvg(
        checkpoint_dir=args.checkpoint_dir,
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=args.num_clients,
        min_evaluate_clients=0,
        min_available_clients=args.num_clients,
        evaluate_fn=evaluate_fn,
        initial_parameters=initial_parameters,
    )

    log("Flower server starting...")

    fl.server.start_server(
        server_address=args.server_address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )

    hist = {"metrics_centralized": metrics_store}
    with open(args.history_out, "w", encoding="utf-8") as f:
        json.dump(hist, f, indent=2)

    best_acc = max((v for _, v in metrics_store["test_acc"]), default=0.0)
    log("=== Training complete ===")
    log(f"Best centralized test accuracy: {best_acc * 100:.2f}%")
    log(f"Checkpoints saved in: {args.checkpoint_dir}")
    log(f"History saved to: {args.history_out}")


def run_client(args) -> None:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Client {args.cid}: starting on {device}")
    x_train, _, y_train, _, input_dim, num_classes = prepare_data(args.seed, args.test_size)
    partitions = stratified_iid_partition(y_train, args.num_clients, args.seed)

    idx = partitions[args.cid]
    client = SERClient(
        x_local=x_train[idx],
        y_local=y_train[idx],
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=args.hidden_dim,
        local_epochs=args.local_epochs,
        lr=args.lr,
        epsilon=args.epsilon,
        max_norm=args.max_norm,
        batch_size=args.batch_size,
        device=device,
    )

    fl.client.start_client(
        server_address=args.server_address,
        client=client.to_client(),
        max_wait_time=180,
    )
    log(f"Client {args.cid}: disconnected")


def run_orchestrator(args) -> None:
    base_cmd = [sys.executable, "flwr_simulation.py"]

    server_cmd = base_cmd + [
        "--mode",
        "server",
        "--num-clients",
        str(args.num_clients),
        "--rounds",
        str(args.rounds),
        "--local-epochs",
        str(args.local_epochs),
        "--lr",
        str(args.lr),
        "--test-size",
        str(args.test_size),
        "--seed",
        str(args.seed),
        "--epsilon",
        str(args.epsilon),
        "--max-norm",
        str(args.max_norm),
        "--checkpoint-dir",
        args.checkpoint_dir,
        "--history-out",
        args.history_out,
        "--server-address",
        args.server_address,
    ]

    server_proc = subprocess.Popen(server_cmd)
    log("Waiting for Flower server to open its port...")

    host, port_text = args.server_address.split(":")
    port = int(port_text)
    deadline = time.time() + 180
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1):
                break
        except OSError:
            time.sleep(1)
    else:
        server_proc.terminate()
        raise RuntimeError(f"Flower server did not become ready on {args.server_address}")

    log("Server ready. Launching clients...")

    client_threads: List[threading.Thread] = []
    for cid in range(args.num_clients):
        client_args = argparse.Namespace(**vars(args))
        client_args.mode = "client"
        client_args.cid = cid
        thread = threading.Thread(target=run_client, args=(client_args,), daemon=True)
        thread.start()
        client_threads.append(thread)

    for thread in client_threads:
        thread.join()
    server_proc.wait()
    log("Orchestrator finished.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Flower-based federated training for SER")
    parser.add_argument("--mode", choices=["orchestrator", "server", "client"], default="orchestrator")
    parser.add_argument("--cid", type=int, default=0)
    parser.add_argument("--num-clients", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--local-epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--max-norm", type=float, default=10.0)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints_flwr")
    parser.add_argument("--history-out", type=str, default="flwr_history.json")
    parser.add_argument("--server-address", type=str, default="127.0.0.1:8088")
    args = parser.parse_args()

    if args.mode == "server":
        run_server(args)
    elif args.mode == "client":
        run_client(args)
    else:
        run_orchestrator(args)


if __name__ == "__main__":
    main()
