import matplotlib.pyplot as plt

def plot_loss(losses, title="Training Loss"):
    plt.plot(losses)
    plt.title(title)
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.show()
