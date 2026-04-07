import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split
import torch

from emotion_inference import load_flower_model

EMOTIONS = ["neutral", "happy", "sad", "angry", "fearful", "disgusted"]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_test_split(test_size: float, seed: int):
    data = np.load("embeddings.npz")
    x, y = data["X"], data["y"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        stratify=y,
        random_state=seed,
    )

    scaler = joblib.load("scaler.pkl")
    x_test = scaler.transform(x_test)
    return x_test, y_test


def predict(model, x_test: np.ndarray):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.no_grad():
        xt = torch.tensor(x_test, dtype=torch.float32).to(device)
        logits = model(xt)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)
    return preds, probs


def plot_accuracy_curve(history_path: str, out_path: str):
    history = joblib.load(history_path) if history_path.endswith(".pkl") else None
    if history is None:
        import json
        with open(history_path, "r", encoding="utf-8") as f:
            history = json.load(f)

    rounds, acc = zip(*history["metrics_centralized"]["test_acc"])
    acc_pct = [a * 100 for a in acc]

    plt.figure(figsize=(9, 5))
    plt.plot(rounds, acc_pct, marker="o", linewidth=2)
    plt.axhline(85, linestyle="--", color="red", label="Target 85%")
    plt.title("Federated Global Test Accuracy by Round")
    plt.xlabel("Round")
    plt.ylabel("Accuracy (%)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_loss_curve(history_path: str, out_path: str):
    import json
    with open(history_path, "r", encoding="utf-8") as f:
        history = json.load(f)

    rounds, loss = zip(*history["metrics_centralized"]["test_loss"])

    plt.figure(figsize=(9, 5))
    plt.plot(rounds, loss, marker="o", linewidth=2, color="darkorange")
    plt.title("Federated Global Test Loss by Round")
    plt.xlabel("Round")
    plt.ylabel("Cross-Entropy Loss")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_confusion(y_true, y_pred, out_path: str):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=EMOTIONS)
    fig, ax = plt.subplots(figsize=(8, 7))
    disp.plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation=25)
    plt.title("Confusion Matrix on Holdout Test Set")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_class_accuracy(y_true, y_pred, out_path: str):
    cm = confusion_matrix(y_true, y_pred)
    class_acc = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)

    plt.figure(figsize=(8, 5))
    plt.bar(EMOTIONS, class_acc * 100, color="teal")
    plt.ylim(0, 100)
    plt.ylabel("Accuracy (%)")
    plt.title("Per-Class Accuracy")
    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def write_metrics(y_true, y_pred, history_path: str, out_path: str):
    import json
    with open(history_path, "r", encoding="utf-8") as f:
        history = json.load(f)

    best_acc = max(v for _, v in history["metrics_centralized"]["test_acc"])
    final_acc = history["metrics_centralized"]["test_acc"][-1][1]

    report = classification_report(y_true, y_pred, target_names=EMOTIONS, digits=4)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("SER Federated Training Metrics\n")
        f.write("=" * 40 + "\n")
        f.write(f"Best test accuracy: {best_acc * 100:.2f}%\n")
        f.write(f"Final test accuracy: {final_acc * 100:.2f}%\n\n")
        f.write("Classification report\n")
        f.write(report)


def main():
    out_dir = "presentation_assets"
    ensure_dir(out_dir)

    model, ckpt_path, hidden_dim = load_flower_model(checkpoint_dir="checkpoints_flwr")
    x_test, y_test = load_test_split(test_size=0.2, seed=42)
    y_pred, _ = predict(model, x_test)

    plot_accuracy_curve("flwr_history.json", os.path.join(out_dir, "accuracy_curve.png"))
    plot_loss_curve("flwr_history.json", os.path.join(out_dir, "loss_curve.png"))
    plot_confusion(y_test, y_pred, os.path.join(out_dir, "confusion_matrix.png"))
    plot_class_accuracy(y_test, y_pred, os.path.join(out_dir, "class_accuracy.png"))
    write_metrics(y_test, y_pred, "flwr_history.json", os.path.join(out_dir, "metrics_summary.txt"))

    print("Generated assets in presentation_assets/")
    print(f"Checkpoint used: {ckpt_path}")
    print(f"Model hidden_dim: {hidden_dim}")


if __name__ == "__main__":
    main()
