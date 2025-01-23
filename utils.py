import matplotlib.pyplot as plt
import numpy as np

def plot_loss(losses, window=50):
    plt.figure(figsize=(10, 5))
    smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
    plt.plot(smoothed)
    plt.title("Training Loss (Smoothed)")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig("training_loss.png")
    plt.close()