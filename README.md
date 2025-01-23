![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green)
[![author](https://img.shields.io/badge/author-mohd--faizy-red)](https://github.com/mohd-faizy)
![Maintained](https://img.shields.io/maintenance/yes/2025)



*A minimal PyTorch implementation of GPT-1 for educational purposes, optimized for Google Colab and CPU training*


## 🛠️ Installation

### Basic Installation
```bash
# Clone repository
git clone https://github.com/mohd-faizy/gpt1-from-scratch.git
cd gpt1-from-scratch

# Install core dependencies
pip install -r requirements.txt
```

### For Google Colab Users

```python
!git clone https://github.com/mohd-faizy/gpt1-from-scratch.git
%cd gpt1-from-scratch

# Install optimized dependencies (-q for quiet mode)
!pip install datasets transformers accelerate -q
```

## 📦 requirements.txt

```txt
torch==2.0.1
transformers==4.30.2
datasets==2.14.4
tqdm==4.65.0
matplotlib==3.7.1
accelerate==0.21.0
```

## 🚀 Quick Start

### 1. Train the Model

```bash
python train.py \
    --batch_size 16 \
    --max_seq_len 128 \
    --epochs 3
```

### 2. Generate Text

```bash
python inference.py \
    --prompt "Artificial intelligence" \
    --temperature 0.7 \
    --top_k 50
```

## 📂 Repository Structure

```
.
├── 📄 config.py          # Model configuration ⚙️
├── 📄 dataset.py         # Data pipeline 🗂️
├── 📄 inference.py       # Text generation 💬
├── 📄 model.py           # GPT architecture 🧩
├── 📄 requirements.txt   # Dependencies 📦
├── 📄 train.py           # Training script 🏋️
├── 📄 utils.py           # Visualization 📊
├── 📁 tokenizer/         # Saved tokenizer
├── 📄 gpt_model.pth      # Trained weights
└── 📄 README.md          # Documentation
```

## 🔧 Key Features

### Training Configuration (`config.py`)

```python
CONFIG = SimpleNamespace(
    n_layer=4,              # Transformer layers
    n_head=4,               # Attention heads
    d_model=256,            # Embedding dimension
    batch_size=16,          # Adjust for your hardware
    lr=2e-4,                # Learning rate
    warmup_steps=100,       # LR scheduling
)
```

### Text Generation (`inference.py`)

```python
def generate(
    prompt: str,
    temperature: float = 0.7,  # Control randomness
    top_k: int = 50,           # Top-k sampling
    max_length: int = 100      # Max generation length
):
    # Generation logic
```

## 📊 Monitoring Training

```python
# Plot smoothed training loss
python -c "from utils import plot_loss; plot_loss()"
```

## 🚨 Troubleshooting

| Issue                        | Solution                          |
|------------------------------|-----------------------------------|
| `DatasetNotFoundError`       | Verify dataset name in config.py  |
| `Tokenizer not found`        | Run train.py before inference     |
| `CUDA out of memory`         | Reduce batch size                 |
| `NaN loss values`            | Lower learning rate               |
| `[PAD] tokens in output`     | Adjust temperature/top_k          |

## 🖥️ Google Colab Workflow

```python
# Full setup & training
!git clone https://github.com/mohd-faizy/gpt1-from-scratch.git
%cd gpt1-from-scratch
!pip install datasets transformers accelerate -q
!python train.py --epochs 3 --batch_size 32
```

## 📚 Resources

- [Original GPT Paper](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [HuggingFace Course](https://huggingface.co/learn/nlp-course)

---

## ⚖ ➤ License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## ❤️ Support

If you find this repository helpful, show your support by starring it! For questions or feedback, reach out on [Twitter(`X`)](https://twitter.com/F4izy).

## 🔗Connect with me

➤ If you have questions or feedback, feel free to reach out!!!

[<img align="left" src="https://cdn4.iconfinder.com/data/icons/social-media-icons-the-circle-set/48/twitter_circle-512.png" width="32px"/>][twitter]
[<img align="left" src="https://cdn-icons-png.flaticon.com/512/145/145807.png" width="32px"/>][linkedin]
[<img align="left" src="https://cdn-icons-png.flaticon.com/512/2626/2626299.png" width="32px"/>][Portfolio]

[twitter]: https://twitter.com/F4izy
[linkedin]: https://www.linkedin.com/in/mohd-faizy/
[Portfolio]: https://ai.stackexchange.com/users/36737/faizy?tab=profile

---

<img src="https://github-readme-stats.vercel.app/api?username=mohd-faizy&show_icons=true" width=380px height=200px />

