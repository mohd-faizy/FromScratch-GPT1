import logging
import torch
from transformers import AutoTokenizer
from model import GPT
from config import CONFIG

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("inference.log"), logging.StreamHandler()]
)

@torch.no_grad()
def generate(prompt, model, tokenizer, device, max_length=50):
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    for _ in range(max_length):
        outputs = model(inputs.input_ids)
        next_token = torch.argmax(outputs[:, -1, :], dim=-1)
        inputs.input_ids = torch.cat([inputs.input_ids, next_token.unsqueeze(-1)], dim=-1)
        
        if next_token == tokenizer.eos_token_id:
            break
    
    return tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)

def main():
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained("./tokenizer")
        
        model = GPT(CONFIG, len(tokenizer)).to(device)
        model.load_state_dict(torch.load("gpt_model.pth", map_location=device))
        
        prompt = "Once upon a time"
        generated = generate(prompt, model, tokenizer, device)
        print(f"\nPrompt: {prompt}\nGenerated: {generated[len(prompt):]}")
        
    except Exception as e:
        logging.error(f"Inference failed: {str(e)}")

if __name__ == "__main__":
    main()