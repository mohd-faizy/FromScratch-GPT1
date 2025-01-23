from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader

class GPTDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.texts = [t for t in texts if len(t) > 0]
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding=False,  # Dynamic padding later
            return_tensors=None,
        )
        return {
            'input_ids': tokens['input_ids'],
            'attention_mask': tokens['attention_mask'],
        }

def get_dataloader(tokenizer):
    dataset = load_dataset(CONFIG.dataset_name, split='train[:{}]'.format(CONFIG.subset_size))
    texts = [text for text in dataset['text'] if text.strip()]
    
    gpt_dataset = GPTDataset(texts, tokenizer, CONFIG.max_seq_len)
    
    collate_fn = lambda batch: tokenizer.pad(
        batch,
        padding='longest',
        max_length=CONFIG.max_seq_len,
        return_tensors='pt',
    )
    
    return DataLoader(
        gpt_dataset,
        batch_size=CONFIG.batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        pin_memory=True,
    )