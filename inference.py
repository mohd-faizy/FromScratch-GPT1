import torch
from transformers import AutoTokenizer
from model import GPT
from config import CONFIG

@torch.no_grad()
def generate(prompt, model, tokenizer, device, max_length=50):
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    for _ in range(max_length):
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits = model(inputs.input_ids).logits[:, -1, :]
        
        # Apply temperature and top-k
        logits = logits / CONFIG.temperature
        top_logits, top_indices = logits.topk(CONFIG.top_k)
        probs = torch.nn.functional.softmax(top_logits, dim=-1)
        
        # Sample from top-k
        next_token = top_indices[0, torch.multinomial(probs, 1)]
        
        # Stop if EOS
        if next_token == tokenizer.eos_token_id:
            break
            
        inputs.input_ids = torch.cat([inputs.input_ids, next_token.unsqueeze(0)], dim=-1)
    
    return tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load safely with weights_only
    tokenizer = AutoTokenizer.from_pretrained("./tokenizer/")
    model = GPT(CONFIG, len(tokenizer)).to(device)
    model.load_state_dict(torch.load("./gpt_model.pth", map_location=device, weights_only=True))
    
    prompt = "Once upon a time"
    generated = generate(prompt, model, tokenizer, device)
    print(f"Prompt: {prompt}\nGenerated: {generated}")