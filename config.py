from types import SimpleNamespace

CONFIG = SimpleNamespace(
    # Architecture
    n_layer=4,                # Reduced layers for Colab/CPU
    n_head=4,
    d_model=256,              # Embedding dimension
    d_ff=1024,                # Feedforward dimension
    max_seq_len=128,          # Sequence length
    
    # Training
    batch_size=8,
    lr=2e-4,
    weight_decay=0.1,
    epochs=100,                 # Start with 10 for testing
    warmup_steps=100,
    grad_clip=1.0,
    
    # Dataset
    dataset_name="wikitext",
    dataset_config="wikitext-2-raw-v1",  # Verified dataset
    subset_size=500,          # Reduced subset
    
    # Generation
    temperature=0.7,
    top_k=50,
)