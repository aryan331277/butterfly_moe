import torch
from torch.utils.data import Dataset


class WikiTextDataset(Dataset):
    def __init__(self, split='train', seq_len=128, dataset_name='wikitext-2-raw-v1'):
        from datasets import load_dataset
        from transformers import AutoTokenizer

        self.seq_len = seq_len
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        self.vocab_size = tokenizer.vocab_size  # 50257

        raw = load_dataset('wikitext', dataset_name, split=split)
        texts = [t for t in raw['text'] if t.strip()]

        all_ids = []
        for text in texts:
            all_ids.extend(tokenizer.encode(text, add_special_tokens=False))

        self.data = []
        for i in range(0, len(all_ids) - seq_len, seq_len):
            chunk = torch.tensor(all_ids[i:i + seq_len + 1], dtype=torch.long)
            self.data.append(chunk)

        print(f"  [{dataset_name} / {split}] {len(self.data)} sequences, vocab={self.vocab_size}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        return seq[:-1], seq[1:]
