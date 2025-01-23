# Import necessary libraries
import logging                          # For recording program execution messages
import torch                            # Main PyTorch library
from transformers import AutoTokenizer  # For text tokenization
from model import GPT                   # Our custom GPT model
from config import CONFIG               # Configuration parameters

# Set up logging to track program execution
logging.basicConfig(
    level=logging.INFO,                 # Show informational messages and above
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
    handlers=[
        logging.FileHandler("inference.log"),  # Save logs to file
        logging.StreamHandler()                # Also show logs in console
    ]
)

# This decorator disables gradient calculation for faster inference
@torch.no_grad()
def generate(prompt, model, tokenizer, device, max_length=50):
    """
    Generates text continuation based on a starting prompt
    Args:
        prompt: Starting text for generation
        model: Our trained GPT model
        tokenizer: Converts text to numbers
        device: CPU or GPU
        max_length: Maximum number of tokens to generate
    """
    model.eval()  # Set model to evaluation mode
    
    # Convert text prompt to numerical tokens
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Generate tokens one by one
    for _ in range(max_length):
        # Get model predictions
        outputs = model(inputs.input_ids)
        
        # Get logits (raw predictions) for last token
        logits = outputs[:, -1, :] / CONFIG.temperature  # Temperature controls randomness
        
        # Top-k sampling: keep only top K most likely tokens
        values, _ = torch.topk(logits, CONFIG.top_k)
        logits[logits < values[:, -1]] = -float('Inf')  # Mask other tokens
        
        # Convert logits to probabilities using softmax
        probs = torch.nn.functional.softmax(logits, dim=-1)
        
        # Randomly sample from top probabilities
        next_token = torch.multinomial(probs, num_samples=1)
        
        # Add new token to existing sequence
        inputs.input_ids = torch.cat([inputs.input_ids, next_token], dim=-1)
        
        # Stop if we generate the end-of-sequence token
        if next_token == tokenizer.eos_token_id:
            break
    
    # Convert numerical tokens back to text
    return tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)

def main():
    """Main function to run text generation"""
    try:
        # Check if GPU is available, use CPU otherwise
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load our custom-trained tokenizer
        tokenizer = AutoTokenizer.from_pretrained("./tokenizer")
        
        # Initialize model architecture
        model = GPT(CONFIG, len(tokenizer)).to(device)
        # Load trained model weights
        model.load_state_dict(torch.load("gpt_model.pth", map_location=device, weights_only=True))
        
        # Example dialogue prompt
        prompt = "The French Revolution began in 1789 and was a period of radical social and political upheaval. The main causes were:"
        # Generate continuation
        generated = generate(prompt, model, tokenizer, device)
        
        # Display results (remove original prompt from output)
        print(f"\nPrompt: {prompt}\nGenerated: {generated[len(prompt):]}")
        
    except Exception as e:
        # Log any errors that occur
        logging.error(f"Inference failed: {str(e)}")

# This ensures the code only runs when executed directly
if __name__ == "__main__":
    main()