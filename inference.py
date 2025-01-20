import torch
from transformers import AutoTokenizer
from model import GPT1  # Ensure GPT1 is correctly implemented
from config import CONFIG


def generate_text(model, tokenizer, prompt, max_length=50, device="cpu"):
    """
    Generates text using the trained model.
    """
    model.eval()
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

    for _ in range(max_length):
        with torch.no_grad():
            outputs = model(input_ids)
            next_token_logits = outputs[:, -1, :]  # Get logits for the last token
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Stop generation if the end-of-text token is generated
            if tokenizer.decode(next_token[0]) == tokenizer.eos_token:
                break

    return tokenizer.decode(input_ids[0])


def main():
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("./tokenizer/")

    # Load trained model
    model = GPT1(
        vocab_size=len(tokenizer),
        max_seq_len=CONFIG["max_seq_len"],
        embedding_dim=CONFIG["embedding_dim"],
        num_heads=CONFIG["num_heads"],
        num_layers=CONFIG["num_layers"],
        hidden_dim=CONFIG["hidden_dim"]
    ).to(device)
    model.load_state_dict(torch.load("./gpt1_model.pth", map_location=device))

    # Add special tokens if necessary
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # Input prompt for generation
    prompt = "Once upon a time"
    print(f"Prompt: {prompt}")

    # Generate text
    generated_text = generate_text(model, tokenizer, prompt, max_length=50, device=device)
    print(f"Generated Text: {generated_text}")


if __name__ == "__main__":
    main()
