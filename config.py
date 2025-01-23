from types import SimpleNamespace

CONFIG = SimpleNamespace(
    # Architecture
    n_layer=6,                # Reduced layers for Colab
    n_head=6,                 # Fewer heads
    d_model=512,              # Reduced embedding dim
    d_ff=2048,                # Feedforward dimension
    max_seq_len=256,          # Increased but manageable
    
    # Training
    batch_size=16,            # Optimized for Colab memory
    lr=3e-4,                  # Slightly higher for faster convergence
    weight_decay=0.1,         # Regularization
    epochs=10,                # Fewer epochs with better data
    warmup_steps=1000,        # Learning rate warmup
    grad_clip=1.0,            # Gradient clipping
    
    # Data
    dataset_name="wikitext-103",  # Larger dataset
    subset_size=5000,         # Manageable subset for Colab
    
    # Generation
    temperature=0.7,          # Sampling temperature
    top_k=50,                 # Top-k sampling
)