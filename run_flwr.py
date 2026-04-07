import argparse
import subprocess
import sys
import time


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch Flower simulation for SER")
    parser.add_argument("--num-clients", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--local-epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--max-norm", type=float, default=10.0)
    parser.add_argument("--server-address", type=str, default="127.0.0.1:8088")
    args = parser.parse_args()

    cmd = [
        sys.executable,
        "flwr_simulation.py",
        "--num-clients",
        str(args.num_clients),
        "--rounds",
        str(args.rounds),
        "--local-epochs",
        str(args.local_epochs),
        "--lr",
        str(args.lr),
        "--batch-size",
        str(args.batch_size),
        "--hidden-dim",
        str(args.hidden_dim),
        "--test-size",
        str(args.test_size),
        "--epsilon",
        str(args.epsilon),
        "--max-norm",
        str(args.max_norm),
        "--server-address",
        args.server_address,
    ]

    start_time = time.perf_counter()
    subprocess.run(cmd, check=True)
    elapsed = time.perf_counter() - start_time
    print(f"Flower training completed in {elapsed:.2f} seconds.")


if __name__ == "__main__":
    main()
