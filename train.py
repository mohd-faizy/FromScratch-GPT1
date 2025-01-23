import logging
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from model import GPT
from config import CONFIG
from dataset import get_dataloader
from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("training.log"), logging.StreamHandler()]
)

def get_lr_scheduler(optimizer, total_steps):
    return LambdaLR(optimizer, lambda step: min(step/CONFIG.warmup_steps, 1))

def train():
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {device}\nConfig: {vars(CONFIG)}")

        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        dataloader = get_dataloader(tokenizer)
        
        model = GPT(CONFIG, len(tokenizer)).to(device)
        logging.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        optimizer = AdamW(model.parameters(), lr=CONFIG.lr, weight_decay=CONFIG.weight_decay)
        total_steps = len(dataloader) * CONFIG.epochs
        scheduler = get_lr_scheduler(optimizer, total_steps)

        model.train()
        for epoch in range(CONFIG.epochs):
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{CONFIG.epochs}", disable=(epoch+1) % 10 != 0)
            for batch in pbar:
                inputs = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                
                outputs = model(inputs, attention_mask=~mask.bool())
                shifted_outputs = outputs[:, :-1, :].contiguous().view(-1, outputs.size(-1))
                shifted_targets = inputs[:, 1:].contiguous().view(-1)
                loss = torch.nn.functional.cross_entropy(
                    shifted_outputs,
                    shifted_targets,
                    ignore_index=tokenizer.pad_token_id,
                )
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG.grad_clip)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                
                pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

            # Log every 10 epochs
            if (epoch + 1) % 10 == 0:
                logging.info(f"Epoch {epoch+1}/{CONFIG.epochs} completed")

        torch.save(model.state_dict(), "gpt_model.pth")
        tokenizer.save_pretrained("./tokenizer")
        logging.info("Training completed!")

    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    train()