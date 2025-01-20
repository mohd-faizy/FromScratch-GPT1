import torch
from transformers import AutoTokenizer
from model import GPT1
from config import CONFIG

def load_model():
    # Load the trained model
    model = GPT1(
        vocab_size=CONFIG["vocab_size"],
        max_seq_len=CONFIG["max_seq_len"],
        embedding_dim=CONFIG["embedding_dim"],
        num_heads=CONFIG["num_heads"],
        num_layers=CONFIG["num_layers"],
        hidden_dim=CONFIG["hidden_dim"]
    )
    model.load_state_dict(torch.load("./gpt1_model.pth"))
    model.eval()
    return model

# def generate_text(prompt, model, tokenizer, max_length=50):
#     model.eval()
#     input_ids = tokenizer.encode(prompt, return_tensors="pt")
#     generated_ids = input_ids

#     for _ in range(max_length):
#         with torch.no_grad():
#             outputs = model(generated_ids)
#             next_token_logits = outputs[:, -1, :]
#             next_token_id = torch.argmax(next_token_logits, dim=-1)
#             generated_ids = torch.cat([generated_ids, next_token_id.unsqueeze(0)], dim=1)

#             # Stop if EOS token is generated
#             if next_token_id == tokenizer.eos_token_id:
#                 break

#     generated_text = tokenizer.decode(generated_ids.squeeze().tolist(), skip_special_tokens=True)
#     return generated_text

import torch.nn.functional as F

def generate_text_with_sampling(prompt, model, tokenizer, max_length=50, temperature=1.0, top_k=50):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    generated_ids = input_ids

    for _ in range(max_length):
        with torch.no_grad():
            outputs = model(generated_ids)
            next_token_logits = outputs[:, -1, :] / temperature
            top_k_logits = torch.topk(next_token_logits, top_k).values
            probabilities = F.softmax(top_k_logits, dim=-1)
            next_token_id = torch.multinomial(probabilities, num_samples=1)
            generated_ids = torch.cat([generated_ids, next_token_id.unsqueeze(0)], dim=1)

            # Stop if EOS token is generated
            if next_token_id == tokenizer.eos_token_id:
                break

    generated_text = tokenizer.decode(generated_ids.squeeze().tolist(), skip_special_tokens=True)
    return generated_text


if __name__ == "__main__":
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("./tokenizer/")
    model = load_model()

    # Provide a prompt and generate text
    prompt = "Once upon a time"
    generated_text = generate_text(prompt, model, tokenizer)
    print(f"Generated Text:\n{generated_text}")
