import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from dataset import get_dataset, TextDataset
from model import GPT1
from config import CONFIG
from utils import plot_loss

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Load and preprocess the dataset
    raw_dataset = get_dataset()
    texts = raw_dataset["text"][:10000]  # Use a subset for quick training
    dataset = TextDataset(texts, tokenizer, CONFIG["max_seq_len"])
    dataloader = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=True)

    # Initialize the model
    model = GPT1(
        vocab_size=CONFIG["vocab_size"],
        max_seq_len=CONFIG["max_seq_len"],
        embedding_dim=CONFIG["embedding_dim"],
        num_heads=CONFIG["num_heads"],
        num_layers=CONFIG["num_layers"],
        hidden_dim=CONFIG["hidden_dim"]
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"])
    loss_fn = torch.nn.CrossEntropyLoss()

    # Training loop
    losses = []
    model.train()
    for epoch in range(CONFIG["epochs"]):
        for step, batch in enumerate(dataloader):
            inputs = batch.to(device)
            labels = batch.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs.view(-1, CONFIG["vocab_size"]), labels.view(-1))
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            if step % 10 == 0:
                print(f"Epoch {epoch}, Step {step}, Loss: {loss.item()}")

    # Plot the loss curve
    plot_loss(losses)

    # Save the trained model and tokenizer
    model_save_path = "./gpt1_model.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    tokenizer_save_path = "./tokenizer/"
    tokenizer.save_pretrained(tokenizer_save_path)
    print(f"Tokenizer saved to {tokenizer_save_path}")

if __name__ == "__main__":
    train()
