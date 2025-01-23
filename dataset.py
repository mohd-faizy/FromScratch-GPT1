# Import necessary libraries
from datasets import load_dataset                 # Hugging Face datasets library for easy data loading
from transformers import AutoTokenizer            # For loading pretrained tokenizers
from torch.utils.data import Dataset, DataLoader  # PyTorch utilities for data handling
from config import CONFIG                         # Import our configuration from previous file

class GPTDataset(Dataset):
    """Custom dataset class for handling text data for GPT-style models"""
    
    def __init__(self, texts, tokenizer, max_length):
        # Initialize dataset with list of texts, tokenizer, and max sequence length
        self.tokenizer = tokenizer  # Converts text to numbers (tokens)
        self.texts = [t for t in texts if t.strip()]  # Filter out empty/whitespace-only texts
        self.max_length = max_length  # Maximum allowed sequence length

    def __len__(self):
        # Return total number of texts in dataset
        return len(self.texts)
    
    def __getitem__(self, idx):
        # Process one text sample at given index
        text = self.texts[idx]
        
        # Tokenize the text with these parameters:
        tokens = self.tokenizer(
            text,
            max_length=self.max_length,  # Truncate if longer than max_length
            truncation=True,             # Enable truncation
            padding='max_length',        # Pad shorter sequences to max_length
            return_tensors=None,         # Return regular Python lists (not tensors)
        )
        
        # Return dictionary containing:
        return {
            'input_ids': tokens['input_ids'],            # The tokenized text as numbers
            'attention_mask': tokens['attention_mask'],  # Mask showing which tokens are real (1) vs padding (0)
        }

def get_dataloader(tokenizer):
    """Create and return a DataLoader for training"""
    
    # Load the dataset from Hugging Face datasets
    dataset = load_dataset(
        CONFIG.dataset_name,                    # From config: "wikitext"
        CONFIG.dataset_config,                  # From config: "wikitext-2-raw-v1"
        split=f'train[:{CONFIG.subset_size}]'   # Take first N samples (500 from config)
    )
    
    # Extract non-empty texts from the dataset
    texts = [text for text in dataset['text'] if text.strip()]
    
    # Safety check: Ensure we have valid texts
    if not texts:
        raise ValueError("No valid texts found in dataset!")
    
    # Create and return the DataLoader with these settings:
    return DataLoader(
        GPTDataset(texts, tokenizer, CONFIG.max_seq_len),  # Our custom dataset
        batch_size=CONFIG.batch_size,                      # From config: 8 samples per batch
        collate_fn=lambda batch: tokenizer.pad(            # How to combine multiple samples
            batch,
            padding='longest',                             # Pad to longest in batch (but we already padded to max_length)
            max_length=CONFIG.max_seq_len,                 # Maximum sequence length from config
            return_tensors='pt',                           # Return PyTorch tensors
        ),
        shuffle=True,                                      # Shuffle data each epoch (good for training)
        num_workers=0                                      # Number of subprocesses for loading data (0=main process)
    )

# This code creates a pipeline that:
# 1. Loads a text dataset from Hugging Face
# 2. Cleans it (removes empty texts)
# 3. Tokenizes the text into numbers the model can understand
# 4. Packages it into batches for efficient training
# 5. Handles padding/truncation to make all sequences the same length
# The DataLoader will feed batches to the model during training