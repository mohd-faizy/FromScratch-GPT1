# Import visualization and math libraries
import matplotlib.pyplot as plt  # For creating plots
import numpy as np  # For numerical operations

def plot_loss(losses, window=50):
    """
    Visualizes training loss with smoothing to better see trends
    Args:
        losses: List of loss values from training
        window: Number of values to average for smoothing (default: 50)
    """
    # Create a figure with specified size (width: 10 inches, height: 5 inches)
    plt.figure(figsize=(10, 5))
    
    # Smooth the loss curve using moving average
    # This helps reduce noise and see the overall trend
    smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
    
    # Plot the smoothed loss values
    plt.plot(smoothed)
    
    # Add chart labels and formatting
    plt.title("Training Loss (Smoothed)")  # Chart title
    plt.xlabel("Steps")  # X-axis label (training iterations)
    plt.ylabel("Loss")   # Y-axis label (error value)
    plt.grid(True)       # Show grid lines for better readability
    
    # Save the plot as an image file
    plt.savefig("training_loss.png")  # Save to current directory
    plt.close()  # Close the figure to free memory
    
    # Try to display in Colab/Jupyter notebooks (optional)
    try:
        from IPython import display
        display.display(display.Image('training_loss.png'))
    except:
        pass  # Skip if not in notebook environment
