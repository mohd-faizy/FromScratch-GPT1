# Import necessary libraries
import logging                                 # For recording training progress
import torch                                   # Main PyTorch library
from torch.optim import AdamW                  # Optimizer for training
from torch.optim.lr_scheduler import LambdaLR  # Learning rate scheduling
from tqdm import tqdm                          # For progress bars
from model import GPT                          # Our GPT model architecture
from config import CONFIG                      # Training configuration
from dataset import get_dataloader             # Data loading function
from transformers import AutoTokenizer         # Text tokenization
from utils import plot_loss                    # Visualization helper

# Set up logging to track training progress
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),  # Save logs to file
        logging.StreamHandler()  # Show logs in console
    ]
)

def get_lr_scheduler(optimizer, total_steps):
    """Create learning rate scheduler with warmup"""
    return LambdaLR(optimizer, lambda step: min(step/CONFIG.warmup_steps, 1))

def train():
    """Main training function"""
    try:
        # Set up device (GPU if available)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {device}\nConfig: {vars(CONFIG)}")

        # Initialize tokenizer and dataset
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token  # Use end-of-sequence as padding token
        dataloader = get_dataloader(tokenizer)  # Get our data loader
        
        # Initialize model
        model = GPT(CONFIG, len(tokenizer)).to(device)
        logging.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Set up optimizer and learning rate scheduler
        optimizer = AdamW(model.parameters(), 
                         lr=CONFIG.lr, 
                         weight_decay=CONFIG.weight_decay)
        total_steps = len(dataloader) * CONFIG.epochs
        scheduler = get_lr_scheduler(optimizer, total_steps)

        # Track loss during training
        all_losses = []
        
        # Start training loop
        model.train()
        for epoch in range(CONFIG.epochs):
            # Progress bar (only shows every 10th epoch)
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{CONFIG.epochs}", 
                       disable=(epoch+1) % 10 != 0)
            
            for batch in pbar:
                # Prepare batch data
                inputs = batch['input_ids'].to(device)  # Input token IDs
                mask = batch['attention_mask'].to(device)  # Attention mask
                
                # Forward pass
                outputs = model(inputs, attention_mask=~mask.bool())
                
                # Shift outputs and targets for next-token prediction
                shifted_outputs = outputs[:, :-1, :].contiguous().view(-1, outputs.size(-1))
                shifted_targets = inputs[:, 1:].contiguous().view(-1)
                
                # Calculate loss (ignore padding tokens)
                loss = torch.nn.functional.cross_entropy(
                    shifted_outputs,
                    shifted_targets,
                    ignore_index=tokenizer.pad_token_id,
                )
                
                # Store loss for visualization
                all_losses.append(loss.item())
                
                # Backpropagation
                loss.backward()
                
                # Prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG.grad_clip)
                
                # Update model weights
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()  # Update learning rate
                
                # Update progress bar
                pbar.set_postfix(loss=f"{loss.item():.4f}", 
                               lr=f"{scheduler.get_last_lr()[0]:.2e}")

            # Log every 10 epochs
            if (epoch + 1) % 10 == 0:
                logging.info(f"Epoch {epoch+1}/{CONFIG.epochs} completed")

        # Save training results
        plot_loss(all_losses)  # Create loss curve
        logging.info("Training loss plot saved to training_loss.png")
        
        torch.save(model.state_dict(), "gpt_model.pth")  # Save model weights
        tokenizer.save_pretrained("./tokenizer")  # Save tokenizer
        logging.info("Training completed!")

    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        raise

# Start training when script is run
if __name__ == "__main__":
    train()