import matplotlib.pyplot as plt
def plot_loss(losses):
    plt.plot(losses)
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.show()
