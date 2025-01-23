from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from config import CONFIG

class GPTDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.texts = [t for t in texts if t.strip()]
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        return {
            'input_ids': tokens['input_ids'],
            'attention_mask': tokens['attention_mask'],
        }

def get_dataloader(tokenizer):
    dataset = load_dataset(
        CONFIG.dataset_name,
        CONFIG.dataset_config,
        split=f'train[:{CONFIG.subset_size}]'
    )
    texts = [text for text in dataset['text'] if text.strip()]
    
    if not texts:
        raise ValueError("No valid texts found in dataset!")
    
    return DataLoader(
        GPTDataset(texts, tokenizer, CONFIG.max_seq_len),
        batch_size=CONFIG.batch_size,
        collate_fn=lambda batch: tokenizer.pad(
            batch,
            padding='longest',
            max_length=CONFIG.max_seq_len,
            return_tensors='pt',
        ),
        shuffle=True,
        num_workers=0  # Essential for Windows
    )