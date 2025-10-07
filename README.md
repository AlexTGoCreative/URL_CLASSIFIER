# DomURLs_BERT Transformers - URL Detection

This repository provides a comprehensive implementation of transformer-based malicious URL detection using **DomURLs_BERT** and **URLBert** models. It includes both pure transformer approaches and hybrid models combining BERT embeddings with handcrafted features.

## Features

- **Multiple modeling approaches**: 
  - Pure transformer models (DomURLs_BERT, URLBert)
  - Hybrid models (BERT embeddings + Random Forest with handcrafted features)
- **URL-focused**: Optimized specifically for malicious URL detection and classification
- **Streamlined codebase**: Removed CNN, RNN, and domain classification components
- **Python 3.11.9 compatible**: Updated dependencies for latest Python version
- **Multiple datasets**: Includes 7+ URL classification datasets (including EBBU dataset)
- **Inference ready**: Complete prediction script for deployed models
- **CPU/GPU support**: Flexible device configuration for training and inference

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

1. **EBBU** - Phishing vs Legitimate URLs (JSON format)
2. **Grambedding_dataset** - Grammar-based URL features
3. **kaggle_malicious_urls** - Kaggle malicious URL dataset
4. **LNU_Phish** - Phishing URL dataset
5. **Mendeley_AK_Singh_2020_phish** - Academic phishing URL dataset
6. **PhishCrawl** - Crawled phishing URLs
7. **PhiUSIIL** - University phishing dataset
8. **ThreatFox_MalURLs** - Threat intelligence malicious URLs

## Usage

### 1. Training Pure Transformer Models

Train transformer-only models (fine-tuning BERT for classification):

```bash
# GPU training
python main_url_transformers.py --dataset Mendeley_AK_Singh_2020_phish --pretrained_path amahdaouy/DomURLs_BERT --num_workers 4 --dropout_prob 0.2 --lr 1e-5 --weight_decay 1e-3 --epochs 10 --batch_size 128 --label_column label --seed 3407 --device 0

# CPU training
python main_url_transformers.py --dataset Mendeley_AK_Singh_2020_phish --pretrained_path amahdaouy/DomURLs_BERT --num_workers 4 --dropout_prob 0.2 --lr 1e-5 --weight_decay 1e-3 --epochs 10 --batch_size 128 --label_column label --seed 3407 --device -1
```

**Parameters:**
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
- `--device`: GPU device ID (use -1 for CPU)

### 2. Training Hybrid Models (BERT + Random Forest)

Train hybrid models combining BERT embeddings with handcrafted features on EBBU dataset:

```bash
# Full dataset training
python main_ebbu_hybrid.py --pretrained_path amahdaouy/DomURLs_BERT --batch_size 32 --n_estimators 100 --max_depth 20 --seed 3407

# Quick test with sample
python main_ebbu_hybrid.py --sample_size 1000 --batch_size 16

# CPU mode (for testing)
python main_ebbu_hybrid.py --sample_size 500 --batch_size 8
```

**Hybrid Model Parameters:**
- `--data_path`: Path to EBBU dataset (default: 'data/url_datasets/EBBU')
- `--pretrained_path`: Pretrained BERT model for embeddings
- `--max_length`: Max sequence length for BERT (default: 128)
- `--batch_size`: Batch size for embedding extraction
- `--n_estimators`: Number of trees in Random Forest (default: 100)
- `--max_depth`: Max depth of Random Forest trees (default: 20)
- `--sample_size`: Sample size for testing (0 = full dataset)
- `--seed`: Random seed

**Features extracted by hybrid model:**
- **17 handcrafted features**: domain_length, domain_randomness, subdomain_count, path_depth, URL_length, digit_count, HTTPS, Alexa ranking, etc.
- **768 BERT embeddings**: Semantic representations from DomURLs_BERT
- **Total: 785 features** combined for Random Forest classification

### 3. Inference with Trained Models

Use trained hybrid models to predict URLs:

```bash
python predict_url.py
```

The script will:
1. Automatically load the most recent trained hybrid model
2. Run predictions on hardcoded test URLs
3. Display detailed analysis with confidence scores and key indicators

**Example output:**
```
🚨 PHISHING
Confidence: 98.45%

Class Probabilities:
       legit: ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  1.55%
       phish: ██████████████████████████████████████████████  98.45%

Key Indicators:
  • Domain Randomness: 0.85 ⚠ HIGH
  • URL Length: 52 chars
  • HTTPS: ✗ No
  • Alexa Ranked: ✗ No
  • Has IP in domain: ✗ No
```

**Customize test URLs:**
Edit `predict_url.py` and modify the `test_urls` list (around line 253):
```python
test_urls = [
    "https://www.google.com",
    "http://suspicious-site.tk/login.php",
    # Add your URLs here
]
```

**Use in Python code:**
```python
from predict_url import PhishingDetector

detector = PhishingDetector(model_path="path/to/model.pkl")
result = detector.predict("https://example.com")

print(f"Verdict: {result['prediction']}")
print(f"Confidence: {result['confidence']*100:.2f}%")
print(f"Is Phishing: {result['is_phishing']}")
```

## Supported Models

### 1. DomURLs_BERT (`amahdaouy/DomURLs_BERT`)
- Pre-trained BERT model specifically for URLs and domains
- Hosted on Hugging Face Hub (auto-downloads)
- Includes URL preprocessing and tokenization
- Used in both pure transformer and hybrid approaches

### 2. URLBert (local model)
- Specialized BERT model trained on URL data
- Requires manual download of model file
- Custom vocabulary optimized for URL structures
- Download: [urlBERT.pt](https://drive.google.com/drive/folders/16pNq7C1gYKR9inVD-P8yPBGS37nitE-D?usp=drive_link)
- Place in: `models/urlbert_model/` folder

### 3. Hybrid Model (BERT + Random Forest)
- Combines BERT semantic embeddings with handcrafted URL features
- Uses Random Forest classifier for final prediction
- Best performance on EBBU dataset
- Includes domain reputation (Alexa ranking) and URL structure analysis

## Project Structure

```
DomURLs_BERT_Transformers/
├── main_url_transformers.py    # Pure transformer training (PyTorch Lightning)
├── main_ebbu_hybrid.py         # Hybrid model training (BERT + Random Forest)
├── predict_url.py              # Inference script for trained models
├── requirements.txt            # Python dependencies
├── utils.py                    # Utility functions
├── .gitignore                  # Git ignore rules (venv, models, mlruns)
├── README.md                   # This file
├── data/
│   └── url_datasets/           # URL classification datasets
│       ├── EBBU/               # EBBU dataset (JSON format)
│       │   ├── data_legitimate.json
│       │   ├── data_phishing.json
│       │   └── top-1m.csv      # Alexa Top 1M domains
│       ├── Grambedding_dataset/
│       ├── kaggle_malicious_urls/
│       ├── LNU_Phish/
│       ├── Mendeley_AK_Singh_2020_phish/
│       ├── PhishCrawl/
│       ├── PhiUSIIL/
│       └── ThreatFox_MalURLs/
├── data_utils/
│   ├── bertdataset.py          # BERT dataset class
│   ├── load_data.py            # Data loading utilities
│   ├── utils.py                # URL preprocessing utilities
│   └── urlbert_vocab/          # URLBert vocabulary
│       └── vocab.txt           # Custom URL vocabulary (5000 tokens)
├── models/
│   ├── plm.py                  # DomURLs_BERT model wrapper
│   ├── urlbert.py              # URLBert model implementation
│   └── urlbert_model/          # URLBert model files
│       ├── config.json         # Model configuration
│       ├── vocab.txt           # URLBert vocabulary
│       ├── urlBERT.pt          # Model weights (download separately)
│       └── README.md           # Model documentation
├── module/
│   ├── pl_module.py            # PyTorch Lightning module
│   └── report.py               # Evaluation metrics
└── mlruns/                     # MLflow tracking (auto-generated)
    ├── experiments/            # Experiment metadata
    └── ckpts/                  # Model checkpoints
```

## Key Changes from Original

- **Removed**: Domain datasets and domain classification functionality
- **Removed**: CNN, RNN, and character-based models (CharCNN, CharLSTM, etc.)
- **Removed**: `main_charnn.py` script
- **Added**: Hybrid model approach (BERT + Random Forest + handcrafted features)
- **Added**: EBBU dataset support with JSON format
- **Added**: Inference script (`predict_url.py`) for model deployment
- **Added**: CPU/GPU flexible training support
- **Updated**: Dependencies for Python 3.11.9 compatibility
- **Focused**: URL detection only (phishing vs legitimate)
- **Streamlined**: Simplified codebase with clear separation of concerns

## Model Approaches Comparison

| Approach | Model | Training Time | Features | Best For |
|----------|-------|---------------|----------|----------|
| **Pure Transformer** | BERT fine-tuning | Slower (GPU recommended) | Semantic embeddings only | Large labeled datasets |
| **Hybrid** | BERT + Random Forest | Medium (works on CPU) | BERT embeddings + 17 handcrafted features | Limited data, interpretability |

## Results and Tracking

The framework uses **MLflow** for experiment tracking:
- **Experiments**: `mlruns/experiments/` - Metadata, parameters, metrics
- **Checkpoints**: `mlruns/ckpts/` - Saved models and artifacts
- **Artifacts**: Classification reports, model metadata, feature importance

View experiments:
```bash
mlflow ui
# Open http://localhost:5000 in browser
```

## Quick Start Workflows

### Workflow 1: Train and Deploy Hybrid Model

```bash
# 1. Train hybrid model on EBBU dataset
python main_ebbu_hybrid.py --sample_size 5000 --batch_size 32

# 2. Test predictions on URLs
python predict_url.py

# 3. View results in MLflow
mlflow ui
```

### Workflow 2: Train Pure Transformer Model

```bash
# 1. Train on any dataset with GPU
python main_url_transformers.py --dataset Mendeley_AK_Singh_2020_phish --pretrained_path amahdaouy/DomURLs_BERT --epochs 10 --batch_size 128 --device 0

# 2. Check results
mlflow ui
```

### Workflow 3: CPU-Only Training (No GPU)

```bash
# Train hybrid model on CPU (recommended for testing)
python main_ebbu_hybrid.py --sample_size 1000 --batch_size 8

# Or train transformer on CPU
python main_url_transformers.py --dataset LNU_Phish --pretrained_path amahdaouy/DomURLs_BERT --epochs 5 --batch_size 32 --device -1
```

## Troubleshooting

### Issue: GitHub push protection (AWS secrets in datasets)
**Solution**: The EBBU dataset may contain URLs with embedded credentials. These are part of the malicious URL samples, not real secrets. To push to GitHub, either:
1. Use the provided links to allow the secrets (confirm they're sample data)
2. Make the repository private
3. Add `data/url_datasets/EBBU/` to `.gitignore`

### Issue: Model file too large for Git
**Solution**: URLBert model file (`urlBERT.pt`) is excluded via `.gitignore`. Download separately from Google Drive.

### Issue: CUDA out of memory
**Solution**: Reduce batch size or use CPU mode (`--device -1`)

### Issue: MLflow tracking not working
**Solution**: Ensure `mlruns/` directory exists and has write permissions

## Advanced Usage

### Custom Handcrafted Features

To add custom features to the hybrid model, edit `main_ebbu_hybrid.py` in the `HandcraftedFeatureExtractor.extract_features()` method:

```python
def extract_features(self, url):
    features = {}
    # ... existing features ...
    
    # Add your custom feature
    features['my_custom_feature'] = your_calculation(url)
    
    return features
```

### Using Different BERT Models

Both scripts support any Hugging Face BERT model:

```bash
# Use a different BERT variant
python main_ebbu_hybrid.py --pretrained_path bert-base-uncased

# Use a multilingual BERT
python main_url_transformers.py --dataset EBBU --pretrained_path bert-base-multilingual-cased
```

### Batch Inference

To predict multiple URLs programmatically:

```python
from predict_url import PhishingDetector

detector = PhishingDetector(model_path="path/to/model.pkl")

urls = ["https://url1.com", "https://url2.com", "https://url3.com"]
results = detector.predict_batch(urls)

for result in results:
    print(f"{result['url']}: {result['prediction']} ({result['confidence']:.2%})")
```