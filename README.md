# GPT-1 from Scratch Repository Guide

## Repository Structure

The repository is structured as follows:
```
.
├── config.py          # Configuration file for model hyperparameters
├── dataset.py         # Dataset preparation and loading logic
├── inference.py       # Script for generating text using the trained model
├── model.py           # Definition of the GPT-1 architecture
├── requirements.txt   # List of Python dependencies
├── train.py           # Training script for the GPT-1 model
├── utils.py           # Utility functions (e.g., plotting loss)
├── tokenizer/         # Directory where the trained tokenizer is saved
├── gpt1_model.pth     # File where the trained model is saved (after training)
└── venv/              # Virtual environment directory (optional, if used)
```

* * *

## File Descriptions

### 1. **`config.py`**

Contains a dictionary (`CONFIG`) with all the configurable parameters for training the GPT-1 model, such as:

- Vocabulary size
- Embedding dimensions
- Number of transformer layers and heads
- Learning rate and batch size

### 2. **`dataset.py`**

Defines logic to:

- Load and preprocess the dataset (`get_dataset()` function).
- Convert text into tokenized input sequences (`TextDataset` class).

### 3. **`model.py`**

Contains the implementation of the GPT-1 architecture, including:

- Embedding layers
- Transformer blocks
- Output layers for text generation.

### 4. **`train.py`**

Main script for training the GPT-1 model. It includes:

- Loading the dataset.
- Initializing the model, optimizer, and loss function.
- Training loop.
- Saving the trained model and tokenizer.

### 5. **`inference.py`**

Script for generating text using the trained model. It:

- Loads the saved model and tokenizer.
- Takes a user-defined prompt as input.
- Outputs the generated text.

### 6. **`utils.py`**

Contains helper functions such as:

- `plot_loss`: Plots the training loss curve for visualization.

### 7. **`requirements.txt`**

Specifies all the Python packages required to run the code. Examples include:

- `torch` for PyTorch
- `transformers` for tokenizer and model utilities
- `datasets` for loading datasets

### 8. **`tokenizer/`**

Directory where the trained tokenizer is saved after running `train.py`. It contains files like:

- `tokenizer_config.json`
- `vocab.json`
- `merges.txt`

### 9. **`gpt1_model.pth`**

File where the trained model's state dictionary is saved after training.

* * *

## Installation Instructions

Follow these steps to set up the repository:

1. **Clone the Repository:**
   ```
   git clone https://github.com/mohd-faizy/gpt1-from-scratch.git
   gh repo clone mohd-faizy/gpt1-from-scratch
   ```
2. **Set Up a Virtual Environment (Optional but Recommended):**
   ```
   python -m venv venv
   source venv/bin/activate  # On Linux/Mac
   venv\Scripts\activate    # On Windows
   ```
3. **Install Dependencies:** Use the `requirements.txt` file to install all required Python packages:
   ```
   pip install -r requirements.txt
   ```
4. **Verify Installation:** Check that all required libraries are installed by running:
   ```
   python -m torch --versionpython -m transformers --version
   ```

* * *

## Running Scripts in VS Code

### Step 1: Open the Repository in VS Code

1. Open VS Code and select **File &gt; Open Folder**.
2. Navigate to the cloned repository and open it.

### Step 2: Configure the Python Environment

1. Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac) to open the command palette.
2. Search for and select **Python: Select Interpreter**.
3. Choose the virtual environment (`venv`) if you created one.

### Step 3: Run `train.py`

1. Open the `train.py` file in the editor.
2. Open the terminal in VS Code (`Ctrl+```).
3. Run the training script:

        python train.py
4. After training completes, the trained model and tokenizer will be saved as:
    - `gpt1_model.pth`
    - `tokenizer/`

### Step 4: Run `inference.py`

1. Open the `inference.py` file in the editor.
2. In the terminal, run:

        python inference.py
3. Enter a prompt (e.g., `"Once upon a time"`) to see the generated text output.

* * *

## Example Workflow

1. Clone the repository and set up the environment.
2. Run `train.py` to train the model:

        python train.py

    Output:

        Epoch 0, Step 0, Loss: 5.678...Model saved to ./gpt1_model.pthTokenizer saved to ./tokenizer/
3. Run `inference.py` to generate text:

        python inference.py

    Output:

        Enter your prompt: Once upon a timeGenerated Text: Once upon a time, in a distant kingdom...

* * *

## Troubleshooting

- **Error: `tokenizer not found`**

  - Ensure you run `train.py` first to save the tokenizer and model.
- **Error: Missing dependencies**

  - Check that all dependencies are installed using `pip install -r requirements.txt`.
- **Error: CUDA not available**

  - Ensure your system has a compatible GPU and CUDA installed. Otherwise, training will default to the CPU, which is slower.

* * *

## Additional Notes

- You can modify `config.py` to change model parameters like the number of layers or training epochs.
- For larger datasets, increase the dataset size in `train.py` (e.g., remove `[:10000]` from `texts`).

