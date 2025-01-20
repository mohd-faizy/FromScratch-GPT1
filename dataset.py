import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_seq_len):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = self.tokenizer(
            text,
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokens["input_ids"].squeeze(0)
        return {
            "input_ids": input_ids,
            "labels": input_ids.clone(),  # For causal language modeling
        }


def get_dataset():
    # Example dataset; replace with actual data loader
    return {"text": ["Hello world", "GPT from scratch is fun!", "Let's build something great!"]}
