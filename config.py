from types import SimpleNamespace

CONFIG = SimpleNamespace(
    # Architecture
    n_layer=4,
    n_head=4,
    d_model=256,
    d_ff=1024,
    max_seq_len=128,
    
    # Training
    batch_size=8,
    lr=2e-4,
    weight_decay=0.1,
    epochs=200,
    warmup_steps=100,
    grad_clip=1.0,
    
    # Dataset
    dataset_name="wikitext",
    dataset_config="wikitext-2-raw-v1",
    subset_size=500,
    
    # Generation
    temperature=0.7,
    top_k=50,
)