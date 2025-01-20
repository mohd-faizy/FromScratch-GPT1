from datasets import load_dataset
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.tokenizer(self.texts[idx], truncation=True, padding="max_length", max_length=self.max_len)
        return torch.tensor(encoded["input_ids"], dtype=torch.long)

def get_dataset():
    dataset = load_dataset("openwebtext", trust_remote_code=True)
    return dataset["train"]