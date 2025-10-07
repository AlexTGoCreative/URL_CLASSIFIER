# DomURLs_BERT Transformers - URL Detection Only

This repository provides a streamlined implementation of transformer-based malicious URL detection using **DomURLs_BERT** and **URLBert** models. This version focuses exclusively on URL classification tasks, removing domain-specific functionality and CNN/RNN models.

## Features

- **Transformer-only architecture**: Uses only pre-trained BERT-based models (DomURLs_BERT, URLBert)
- **URL-focused**: Optimized specifically for malicious URL detection and classification
- **Streamlined codebase**: Removed CNN, RNN, and domain classification components
- **Python 3.13.6 compatible**: Updated dependencies for latest Python version
- **Multiple datasets**: Includes 7 URL classification datasets

## Requirements

- Python 3.11.9
- torch 2.5.1
- transformers 4.46.0
- lightning 2.4.0
- mlflow 2.18.0

The complete list of requirements is in `requirements.txt`

## Installation

1. Clone this repository
2. Create a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. For URLBert model, download the [urlBERT.pt](https://drive.google.com/drive/folders/16pNq7C1gYKR9inVD-P8yPBGS37nitE-D?usp=drive_link) model into `models\urlbert_model` folder.

## Available URL Datasets

This repository includes the following URL classification datasets:

1. **Grambedding_dataset** - Grammar-based URL features
2. **kaggle_malicious_urls** - Kaggle malicious URL dataset
3. **LNU_Phish** - Phishing URL dataset
4. **Mendeley_AK_Singh_2020_phish** - Academic phishing URL dataset
5. **PhishCrawl** - Crawled phishing URLs
6. **PhiUSIIL** - University phishing dataset
7. **ThreatFox_MalURLs** - Threat intelligence malicious URLs

## Usage

### Training Transformer Models

Run the main training script with the following parameters:

```bash
python main_url_transformers.py --dataset Mendeley_AK_Singh_2020_phish --pretrained_path amahdaouy/DomURLs_BERT --num_workers 4 --dropout_prob 0.2 --lr 1e-5 --weight_decay 1e-3 --epochs 10 --batch_size 128 --label_column label --seed 3407 --device 0

python main_url_transformers.py --dataset Mendeley_AK_Singh_2020_phish --pretrained_path amahdaouy/DomURLs_BERT --num_workers 4 --dropout_prob 0.2 --lr 1e-5 --weight_decay 1e-3 --epochs 10 --batch_size 128 --label_column label --seed 3407 --device cpu
```

### Parameters

- `--dataset`: URL dataset name from the available datasets
- `--pretrained_path`: Pretrained transformer model path (supports 'amahdaouy/DomURLs_BERT' and URLBert)
- `--num_workers`: Number of workers for data loading
- `--dropout_prob`: Dropout probability for regularization
- `--lr`: Learning rate for optimizer
- `--weight_decay`: Weight decay for regularization
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size for training
- `--label_column`: Name of the label column in dataset
- `--seed`: Random seed for reproducibility
- `--device`: GPU device ID

### Supported Models

1. **DomURLs_BERT** (`amahdaouy/DomURLs_BERT`)
   - Pre-trained BERT model specifically for URLs and domains
   - Includes URL preprocessing and tokenization
   
2. **URLBert** (local model)
   - Specialized BERT model trained on URL data
   - Requires downloading the model file (see installation instructions)

## Project Structure

```
DomURLs_BERT_Transformers/
├── main_url_transformers.py    # Main training script
├── requirements.txt            # Python 3.13.6 compatible dependencies
├── utils.py                   # Utility functions
├── README.md                  # This file
├── data/
│   └── url_datasets/          # URL classification datasets
├── data_utils/
│   ├── bertdataset.py         # BERT dataset class
│   ├── load_data.py           # Data loading utilities
│   ├── utils.py               # URL preprocessing utilities
│   └── urlbert_vocab/         # URLBert vocabulary
├── models/
│   ├── plm.py                 # DomURLs_BERT model
│   ├── urlbert.py             # URLBert model
│   └── urlbert_model/         # URLBert model files
└── module/
    ├── pl_module.py           # PyTorch Lightning module
    └── report.py              # Evaluation metrics
```

## Key Changes from Original

- **Removed**: Domain datasets and domain classification functionality
- **Removed**: CNN, RNN, and character-based models (CharCNN, CharLSTM, etc.)
- **Removed**: `main_charnn.py` script
- **Updated**: Dependencies for Python 3.13.6 compatibility
- **Focused**: Only transformer-based models for URL detection
- **Streamlined**: Simplified codebase with URL-only focus

## Results and Tracking

The framework uses MLflow for experiment tracking. Results are stored in:
- `mlruns/experiments/` - Experiment metadata
- `mlruns/ckpts/` - Model checkpoints