# Import SimpleNamespace to create a simple object for storing configuration values
from types import SimpleNamespace

# Create a configuration object using SimpleNamespace to organize all model settings
CONFIG = SimpleNamespace(
    # ========== Model Architecture Parameters ==========
    # These settings define the structure of the neural network
    
    n_layer=4,          # Number of transformer layers in the model
    n_head=4,           # Number of attention heads in each transformer layer
    d_model=256,        # Dimension of embeddings (size of vectors used in the model)
    d_ff=1024,          # Dimension of feed-forward network's hidden layer
    max_seq_len=128,    # Maximum sequence length the model can process

    # ========== Training Parameters ==========
    # These settings control how the model learns from data
    
    batch_size=8,       # Number of samples processed in one training step
    lr=2e-4,            # Learning rate - how quickly the model updates its weights
    weight_decay=0.1,   # Regularization to prevent overfitting (penalize large weights)
    epochs=150,         # Number of complete passes through the training data
    warmup_steps=100,   # Gradually increase learning rate at start of training
    grad_clip=1.0,      # Prevent exploding gradients by clipping their maximum value

    # ========== Dataset Parameters ==========
    # Settings related to the training data
    
    dataset_name="wikitext",             # Name of the dataset to use
    dataset_config="wikitext-2-raw-v1",  # Specific version of the dataset
    subset_size=500,                     # Optional: Use smaller subset of data for faster testing

    # ========== Text Generation Parameters ==========
    # Settings for how the model generates text
    
    temperature=0.7,    # Controls randomness: lower = more predictable, higher = more creative
    top_k=50,           # Only consider top K most likely next words during generation
)

# This configuration object helps keep all hyperparameters organized in one place
# The values shown are typical starting points for a small transformer model
# As you learn more about AI, you can experiment with adjusting these values!