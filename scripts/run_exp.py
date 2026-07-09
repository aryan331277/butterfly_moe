
import torch
from butterflymoe.data import WikiTextDataset
from butterflymoe.training import MAX_STEPS
from torch.utils.data import DataLoader

from experiments.exp1_accuracy import exp1_main_accuracy_table
from experiments.exp2_scaling import exp2_accuracy_memory_and_diversity
from experiments.exp4_non_pow2 import exp4_non_power_of_two
from experiments.exp6_memory import exp6_memory_scaling


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seq_len = 128
    batch_size = 32
    d_model = 512
    num_layers = 4
    num_experts = 8
    top_k = 2
    num_epochs = 20

    print("Loading WikiText-2...")
    train_ds = WikiTextDataset(split="train", seq_len=seq_len)
    val_ds = WikiTextDataset(split="validation", seq_len=seq_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    vocab_size = train_ds.vocab_size

    # Exp 1: main accuracy table
    results, bmoe, smoe, dense = exp1_main_accuracy_table(
        train_loader, val_loader, vocab_size, device,
        d_model, num_layers, num_experts, top_k, num_epochs,
    )

if __name__ == "__main__":
    main()
