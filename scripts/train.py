
import argparse
import torch
import yaml

from butterflymoe.data import WikiTextDataset
from butterflymoe.models import ButterflyMoE_LM
from butterflymoe.training import train_model
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser(description="Train ButterflyMoE")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    m = cfg["model"]
    t = cfg["training"]
    d = cfg["data"]

    device = torch.device(args.device)

    print("Loading data...")
    train_ds = WikiTextDataset(split="train", seq_len=t["seq_len"], dataset_name=d["dataset_name"])
    val_ds = WikiTextDataset(split="validation", seq_len=t["seq_len"], dataset_name=d["dataset_name"])
    train_loader = DataLoader(train_ds, batch_size=t["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=t["batch_size"])

    print(f"Building ButterflyMoE: d={m['d_model']}, L={m['num_layers']}, "
          f"N={m['num_experts']}, k={m['top_k']}")
    model = ButterflyMoE_LM(
        vocab_size=d["vocab_size"],
        d_model=m["d_model"],
        num_layers=m["num_layers"],
        num_experts=m["num_experts"],
        top_k=m["top_k"],
        num_butterfly_layers=m.get("num_butterfly_layers"),
    ).to(device)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    train_model(
        model, train_loader, val_loader, device,
        num_epochs=t["num_epochs"],
        lr=t["lr"],
        load_balance_weight=t["load_balance_weight"],
        model_name="butterflymoe",
        max_steps=t["max_steps"],
    )


if __name__ == "__main__":
    main()
