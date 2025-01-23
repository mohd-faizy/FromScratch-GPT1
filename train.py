import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from model import GPT
from config import CONFIG
from dataset import get_dataloader
from transformers import AutoTokenizer

def get_lr_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))
    return LambdaLR(optimizer, lr_lambda)

def train():
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 for matmul
    torch.backends.cudnn.allow_tf32 = True  # Enable TF32 for convolutions
    
    # Load components
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    dataloader = get_dataloader(tokenizer)
    
    model = GPT(CONFIG, len(tokenizer)).to(device)
    
    # Optimizer with weight decay
    no_decay = ["bias", "LayerNorm.weight"]
    params = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": CONFIG.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(params, lr=CONFIG.lr)
    
    # Scheduler
    total_steps = len(dataloader) * CONFIG.epochs
    scheduler = get_lr_scheduler(optimizer, CONFIG.warmup_steps, total_steps)
    
    # Mixed precision
    scaler = torch.cuda.amp.GradScaler()
    
    # Training loop
    model.train()
    for epoch in range(CONFIG.epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{CONFIG.epochs}")
        for batch in pbar:
            inputs = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model(inputs, attention_mask=~mask.bool())
                loss = torch.nn.functional.cross_entropy(
                    outputs.view(-1, outputs.size(-1)),
                    inputs.view(-1),
                    ignore_index=tokenizer.pad_token_id,
                )
            
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG.grad_clip)
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            
            pbar.set_postfix(loss=loss.item(), lr=scheduler.get_last_lr()[0])
    
    # Save model safely
    torch.save(model.state_dict(), "./gpt_model.pth", _use_new_zipfile_serialization=True)
    tokenizer.save_pretrained("./tokenizer/")