import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from dataset import TextDataset, get_dataset
from model import GPT1
from config import CONFIG
from utils import plot_loss


def resize_embeddings(model, tokenizer):
    """
    Resizes the embedding layer of the model to match the tokenizer's vocabulary size.
    """
    old_embeddings = model.embedding.weight.data
    new_vocab_size = len(tokenizer)
    new_embeddings = torch.nn.Embedding(new_vocab_size, old_embeddings.size(1)).to(old_embeddings.device)

    # Copy existing weights into the resized embeddings
    new_embeddings.weight.data[:old_embeddings.size(0), :] = old_embeddings
    model.embedding = new_embeddings
    return model


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Add a padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # Load and preprocess the dataset
    raw_dataset = get_dataset()  # Assumes get_dataset() loads your dataset
    texts = raw_dataset["text"][:10000]  # Subset for quick training
    dataset = TextDataset(texts, tokenizer, CONFIG["max_seq_len"])
    dataloader = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=True)

    # Initialize the model
    model = GPT1(
        vocab_size=len(tokenizer),
        max_seq_len=CONFIG["max_seq_len"],
        embedding_dim=CONFIG["embedding_dim"],
        num_heads=CONFIG["num_heads"],
        num_layers=CONFIG["num_layers"],
        hidden_dim=CONFIG["hidden_dim"]
    ).to(device)

    # Resize embeddings
    model = resize_embeddings(model, tokenizer)

    # Define optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"])
    loss_fn = torch.nn.CrossEntropyLoss()

    # Training loop
    losses = []
    model.train()
    for epoch in range(CONFIG["epochs"]):
        for step, batch in enumerate(dataloader):
            inputs = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            # Compute loss
            loss = loss_fn(outputs.view(-1, len(tokenizer)), labels.view(-1))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            if step % 10 == 0:
                print(f"Epoch {epoch}, Step {step}, Loss: {loss.item()}")

    # Plot the loss curve
    plot_loss(losses)

    # Save the model and tokenizer
    torch.save(model.state_dict(), "./gpt1_model.pth")
    tokenizer.save_pretrained("./tokenizer/")
    print("Training complete!")


if __name__ == "__main__":
    train()
