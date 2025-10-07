"""
EBBU Phishing URL Detection with Hybrid Features
Combines DomURLs_BERT embeddings with handcrafted features
Based on kaggle.py implementation adapted to project structure
"""

import mlflow
import torch
import numpy as np
import pandas as pd
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import MLFlowLogger
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, classification_report, confusion_matrix)
from collections import OrderedDict, Counter
from pathlib import Path
from tqdm.auto import tqdm
from urllib.parse import urlparse
import argparse
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# SECTION 1: DATA LOADING
# ============================================================================

def load_ebbu_dataset(data_path='data/url_datasets/EBBU'):
    """
    Load EBBU dataset from JSON files
    
    Returns:
        df_train, df_dev, df_test: DataFrames with 'url' and 'label' columns
        label_encoder: LabelEncoder instance
    """
    print("Loading EBBU dataset...")
    
    # Load legitimate URLs
    with open(f'{data_path}/data_legitimate.json', 'r') as f:
        legitimate_urls = json.load(f)
    
    # Load phishing URLs
    with open(f'{data_path}/data_phishing.json', 'r') as f:
        phishing_urls = json.load(f)
    
    # Create DataFrames
    legitimate_df = pd.DataFrame({'url': legitimate_urls, 'label': 'legit'})
    phishing_df = pd.DataFrame({'url': phishing_urls, 'label': 'phish'})
    
    print(f"Loaded {len(legitimate_df)} legitimate URLs")
    print(f"Loaded {len(phishing_df)} phishing URLs")
    
    # Combine and shuffle
    df = pd.concat([legitimate_df, phishing_df], ignore_index=True)
    df = df.dropna(subset=['url'])
    df = df[df['url'].str.strip() != '']
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Total URLs: {len(df)}")
    print(f"Balance: {(df['label']=='phish').sum() / len(df):.2%} phishing")
    
    # Encode labels
    label_encoder = LabelEncoder()
    df['label_encoded'] = label_encoder.fit_transform(df['label'])
    
    # Split into train/dev/test (70/15/15)
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['label'])
    dev_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])
    
    print(f"\nTrain: {len(train_df)} | Dev: {len(dev_df)} | Test: {len(test_df)}")
    
    return train_df, dev_df, test_df, label_encoder

# ============================================================================
# SECTION 2: HANDCRAFTED FEATURE EXTRACTION (17 Features)
# ============================================================================

class HandcraftedFeatureExtractor:
    """
    Extract 17 handcrafted features to complement BERT embeddings
    Based on Sahingoz et al. 2019 URL analysis research
    
    Features:
    1. Domain Randomness - Statistical randomness of domain name
    2. Is Random Domain - Binary flag for high randomness
    3. Alexa Top 1M - Domain reputation check
    4. Alexa Top 100K - Premium domain reputation
    5. Subdomain Count - Number of subdomains
    6. Domain Length - Length of domain name
    7. Path Length - Length of URL path
    8. Path Depth - Directory depth (/ count)
    9. URL Length - Total URL length
    10. Digit Count - Number of digits in URL
    11. Special Char Count - Special characters count
    12. Has IP - IP address in domain
    13. HTTPS - Secure protocol usage
    14. Has WWW - Starts with www
    15. Punycode - International domain encoding
    16. Consecutive Chars - Max character repetition
    17. Known TLD - TLD in known list
    """
    
    def __init__(self, alexa_path='data/url_datasets/EBBU/top-1m.csv'):
        self.alexa_domains = None
        self.alexa_path = alexa_path
        # Top 20 most common TLDs worldwide
        self.known_tlds = {
            'com', 'org', 'net', 'edu', 'gov', 'mil', 'int',
            'de', 'uk', 'cn', 'nl', 'eu', 'ru', 'br', 'au',
            'fr', 'it', 'ca', 'es', 'pl', 'jp', 'in', 'ch'
        }
    
    def extract_features(self, url):
        """Extract 17 handcrafted features from URL"""
        features = {}
        
        try:
            parsed = urlparse(url)
            domain = parsed.netloc
            path = parsed.path
            query = parsed.query
            
            # Get registered domain (remove subdomains)
            registered_domain = self._get_registered_domain(domain)
            
            # 1-2. Domain Randomness features
            features['domain_randomness'] = self._calculate_domain_randomness(registered_domain)
            features['is_random_domain'] = int(features['domain_randomness'] > 0.7)
            
            # 3-4. Alexa reputation features
            features['alexa_top_1m'] = int(self._is_alexa_domain(domain))
            features['alexa_top_100k'] = int(self._is_alexa_domain(domain, top_n=100000))
            
            # 5-6. Subdomain features
            features['subdomain_count'] = domain.count('.') - 1 if '.' in domain else 0
            features['domain_length'] = len(domain)
            
            # 7-8. Path features
            features['path_length'] = len(path)
            features['path_depth'] = path.count('/')
            
            # 9-10. URL character features
            features['url_length'] = len(url)
            features['digit_count'] = sum(c.isdigit() for c in url)
            
            # 11. Special characters (Sahingoz et al. 2019: '-', '.', '/', '@', '?', '&', '=', '_')
            special_chars = {'-', '.', '/', '@', '?', '&', '=', '_'}
            features['special_char_count'] = sum(c in special_chars for c in url)
            
            # 12. IP address in domain
            features['has_ip'] = int(self._has_ip_address(domain))
            
            # 13. HTTPS protocol
            features['https'] = int(parsed.scheme == 'https')
            
            # 14. WWW prefix
            features['has_www'] = int(domain.startswith('www.'))
            
            # 15. Punycode encoding (internationalized domains)
            features['has_punycode'] = int('xn--' in domain.lower())
            
            # 16. Consecutive character repetition (phishing technique)
            features['max_consecutive_chars'] = self._max_consecutive_chars(url)
            
            # 17. Known TLD
            tld = domain.split('.')[-1] if '.' in domain else ''
            features['known_tld'] = int(tld.lower() in self.known_tlds)
            
        except Exception as e:
            # Return default features on error
            features = {
                'domain_randomness': 0.0, 'is_random_domain': 0,
                'alexa_top_1m': 0, 'alexa_top_100k': 0,
                'subdomain_count': 0, 'domain_length': 0,
                'path_length': 0, 'path_depth': 0,
                'url_length': 0, 'digit_count': 0,
                'special_char_count': 0, 'has_ip': 0,
                'https': 0, 'has_www': 0,
                'has_punycode': 0, 'max_consecutive_chars': 0,
                'known_tld': 0
            }
        
        return features
    
    def _get_registered_domain(self, domain):
        """Extract registered domain (remove subdomains)"""
        if not domain:
            return ''
        if domain.startswith('www.'):
            domain = domain[4:]
        parts = domain.split('.')
        if len(parts) >= 2:
            return '.'.join(parts[-2:])
        return domain
    
    def _calculate_entropy(self, text):
        """Calculate Shannon entropy"""
        if not text:
            return 0
        counter = Counter(text)
        length = len(text)
        entropy = -sum((count/length) * np.log2(count/length) for count in counter.values())
        return entropy
    
    def _calculate_domain_randomness(self, domain):
        """Calculate randomness score for domain"""
        if not domain:
            return 0.0
        domain_name = domain.split('.')[0] if '.' in domain else domain
        entropy = self._calculate_entropy(domain_name)
        max_entropy = np.log2(min(26, len(domain_name))) if len(domain_name) > 0 else 1
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        has_numbers = any(c.isdigit() for c in domain_name)
        length_factor = min(1.0, len(domain_name) / 20)
        randomness_score = (normalized_entropy * 0.7 + 
                          (0.1 if has_numbers else 0) +
                          length_factor * 0.2)
        return min(1.0, randomness_score)
    
    def _has_ip_address(self, domain):
        """Check if domain contains IP address"""
        import re
        ip_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
        return bool(re.search(ip_pattern, domain))
    
    def _max_consecutive_chars(self, text):
        """Calculate maximum consecutive character repetition"""
        if not text or len(text) < 2:
            return 0
        max_count = 1
        current_count = 1
        for i in range(1, len(text)):
            if text[i] == text[i-1]:
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 1
        return max_count
    
    def _load_alexa_domains(self):
        """Load Alexa Top 1M domains"""
        if self.alexa_domains is not None:
            return
        try:
            df = pd.read_csv(self.alexa_path, names=['rank', 'domain'])
            self.alexa_domains = set(df['domain'].str.lower())
            print(f"Loaded {len(self.alexa_domains)} Alexa domains")
        except Exception as e:
            print(f"Warning: Could not load Alexa domains: {e}")
            self.alexa_domains = set()
    
    def _is_alexa_domain(self, domain, top_n=1000000):
        """Check if domain is in Alexa Top N list"""
        if self.alexa_domains is None:
            self._load_alexa_domains()
        registered_domain = self._get_registered_domain(domain)
        return registered_domain.lower() in self.alexa_domains

# ============================================================================
# SECTION 3: BERT EMBEDDING EXTRACTION
# ============================================================================

class DomURLsBERTEmbedder:
    """Extract embeddings using DomURLs_BERT"""
    
    def __init__(self, model_name='amahdaouy/DomURLs_BERT', max_length=128):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        print(f"Loading {model_name} model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.max_length = max_length
        print("Model loaded successfully!")
    
    def get_embedding(self, url, pooling='mean'):
        """Extract embedding for a single URL"""
        try:
            inputs = self.tokenizer(
                url,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                hidden_states = outputs.last_hidden_state
            
            if pooling == 'mean':
                attention_mask = inputs['attention_mask'].unsqueeze(-1)
                embeddings = (hidden_states * attention_mask).sum(1) / attention_mask.sum(1)
            elif pooling == 'cls':
                embeddings = hidden_states[:, 0, :]
            else:
                embeddings = hidden_states.max(1)[0]
            
            return embeddings.cpu().numpy().flatten()
        
        except Exception as e:
            print(f"Error getting embedding for {url}: {e}")
            return np.zeros(768)
    
    def get_embeddings_batch(self, urls, batch_size=32, pooling='mean'):
        """Extract embeddings for multiple URLs in batches"""
        embeddings = []
        for i in tqdm(range(0, len(urls), batch_size), desc="Extracting BERT embeddings"):
            batch = urls[i:i+batch_size]
            batch_embeddings = [self.get_embedding(url, pooling) for url in batch]
            embeddings.extend(batch_embeddings)
        return np.array(embeddings)

# ============================================================================
# SECTION 4: FEATURE COMBINATION
# ============================================================================

def build_feature_matrix(df, embedder, feature_extractor, batch_size=32):
    """Build combined feature matrix"""
    print(f"\nProcessing {len(df)} URLs...")
    
    # Extract handcrafted features
    print("1. Extracting handcrafted features...")
    handcrafted_features = []
    for url in tqdm(df['url'], desc="Handcrafted features"):
        features = feature_extractor.extract_features(url)
        handcrafted_features.append(features)
    
    handcrafted_df = pd.DataFrame(handcrafted_features)
    print(f"   Handcrafted features shape: {handcrafted_df.shape}")
    
    # Extract BERT embeddings
    print("2. Extracting BERT embeddings...")
    bert_embeddings = embedder.get_embeddings_batch(df['url'].tolist(), batch_size=batch_size)
    print(f"   BERT embeddings shape: {bert_embeddings.shape}")
    
    # Combine features
    print("3. Combining features...")
    X_handcrafted = handcrafted_df.values
    X_bert = bert_embeddings
    X = np.hstack([X_handcrafted, X_bert])
    y = df['label_encoded'].values
    
    feature_names = list(handcrafted_df.columns) + [f'bert_dim_{i}' for i in range(bert_embeddings.shape[1])]
    
    print(f"\n   Combined feature matrix shape: {X.shape}")
    print(f"   Total features: {len(feature_names)}")
    print(f"   - Handcrafted: {len(handcrafted_df.columns)}")
    print(f"   - BERT: {bert_embeddings.shape[1]}")
    
    return X, y, feature_names

# ============================================================================
# SECTION 5: MODEL TRAINING
# ============================================================================

def train_hybrid_model(X_train, y_train, X_dev, y_dev, args):
    """Train Random Forest with hybrid features"""
    print("\n" + "="*70)
    print("TRAINING HYBRID MODEL (Random Forest + BERT)")
    print("="*70)
    
    # Normalize features
    print("\nNormalizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_dev_scaled = scaler.transform(X_dev)
    
    # Train Random Forest
    print("\nTraining Random Forest classifier...")
    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=args.seed,
        n_jobs=-1,
        verbose=1
    )
    
    clf.fit(X_train_scaled, y_train)
    print("Training complete!")
    
    # Evaluate
    y_pred_train = clf.predict(X_train_scaled)
    y_pred_dev = clf.predict(X_dev_scaled)
    
    results = {
        'train': {
            'accuracy': accuracy_score(y_train, y_pred_train),
            'precision': precision_score(y_train, y_pred_train, average='weighted'),
            'recall': recall_score(y_train, y_pred_train, average='weighted'),
            'f1': f1_score(y_train, y_pred_train, average='weighted')
        },
        'dev': {
            'accuracy': accuracy_score(y_dev, y_pred_dev),
            'precision': precision_score(y_dev, y_pred_dev, average='weighted'),
            'recall': recall_score(y_dev, y_pred_dev, average='weighted'),
            'f1': f1_score(y_dev, y_pred_dev, average='weighted')
        }
    }
    
    print("\nTRAIN SET:")
    for metric, value in results['train'].items():
        print(f"  {metric.capitalize()}: {value:.4f}")
    
    print("\nDEV SET:")
    for metric, value in results['dev'].items():
        print(f"  {metric.capitalize()}: {value:.4f}")
    
    return clf, scaler, results

# ============================================================================
# SECTION 6: MAIN EXECUTION
# ============================================================================

def main(args):
    """Main training pipeline"""
    
    # Set random seed
    seed_everything(seed=args.seed)
    
    # Load dataset
    print("\n[STEP 1] Loading EBBU dataset...")
    train_df, dev_df, test_df, label_encoder = load_ebbu_dataset(args.data_path)
    num_classes = len(label_encoder.classes_)
    
    # Sample data if requested (for testing)
    if args.sample_size > 0:
        print(f"\nSampling {args.sample_size} URLs for faster testing...")
        train_df = train_df.sample(n=min(args.sample_size, len(train_df)), random_state=args.seed)
        dev_df = dev_df.sample(n=min(args.sample_size//5, len(dev_df)), random_state=args.seed)
        test_df = test_df.sample(n=min(args.sample_size//5, len(test_df)), random_state=args.seed)
    
    # Initialize extractors
    print("\n[STEP 2] Initializing feature extractors...")
    embedder = DomURLsBERTEmbedder(model_name=args.pretrained_path, max_length=args.max_length)
    feature_extractor = HandcraftedFeatureExtractor(alexa_path=f'{args.data_path}/top-1m.csv')
    
    # Build feature matrices
    print("\n[STEP 3] Building feature matrices...")
    X_train, y_train, feature_names = build_feature_matrix(train_df, embedder, feature_extractor, args.batch_size)
    X_dev, y_dev, _ = build_feature_matrix(dev_df, embedder, feature_extractor, args.batch_size)
    X_test, y_test, _ = build_feature_matrix(test_df, embedder, feature_extractor, args.batch_size)
    
    # MLFlow setup
    experiment_params = {
        "lr": "N/A (RandomForest)",
        "epochs": "N/A (RandomForest)",
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "dataset": "EBBU",
        "pretrained_path": args.pretrained_path,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "sample_size": args.sample_size if args.sample_size > 0 else "Full dataset",
        "num_classes": num_classes,
        "model_type": "RandomForest + BERT Hybrid"
    }
    
    mlflow.set_tracking_uri(Path.cwd().joinpath("mlruns/experiments").as_uri())
    exp_name = f'EBBU_hybrid_classification'
    print(f'\nExperiment: {exp_name}')
    mlflow.set_experiment(exp_name)
    mlflow.start_run(run_name=f"RF_BERT_{args.pretrained_path.split('/')[-1]}")
    mlflow.log_params(experiment_params)
    mlflow.set_tags({
        "project_name": "DomURLs_BERT_Transformers",
        "dataset": "EBBU",
        "model_type": "Hybrid (RandomForest + BERT)"
    })
    
    # Train model
    print("\n[STEP 4] Training model...")
    clf, scaler, results = train_hybrid_model(X_train, y_train, X_dev, y_dev, args)
    
    # Test evaluation
    print("\n[STEP 5] Evaluating on test set...")
    X_test_scaled = scaler.transform(X_test)
    y_pred_test = clf.predict(X_test_scaled)
    
    test_results = {
        'accuracy': accuracy_score(y_test, y_pred_test),
        'precision': precision_score(y_test, y_pred_test, average='weighted'),
        'recall': recall_score(y_test, y_pred_test, average='weighted'),
        'f1': f1_score(y_test, y_pred_test, average='weighted')
    }
    
    print("\nTEST SET:")
    for metric, value in test_results.items():
        print(f"  {metric.capitalize()}: {value:.4f}")
    
    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, y_pred_test, target_names=label_encoder.classes_))
    
    # Log metrics to MLFlow
    for split, metrics in [('train', results['train']), ('dev', results['dev']), ('test', test_results)]:
        for metric, value in metrics.items():
            mlflow.log_metric(f"{split}_{metric}", value)
    
    # Save model
    print("\n[STEP 6] Saving model...")
    checkpoint_path = f"./mlruns/ckpts/ebbu_hybrid_{mlflow.active_run().info.run_id}/"
    Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
    
    model_package = {
        'classifier': clf,
        'scaler': scaler,
        'feature_names': feature_names,
        'label_encoder': label_encoder,
        'model_type': 'RandomForest + DomURLs_BERT',
        'experiment_params': experiment_params
    }
    
    model_path = f"{checkpoint_path}/hybrid_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model_package, f)
    
    mlflow.log_artifact(model_path)
    print(f"Model saved to {model_path}")
    
    # Save classification report
    report_path = f"{checkpoint_path}/classification_report.txt"
    with open(report_path, 'w') as f:
        f.write("EBBU Hybrid Model Classification Report\n")
        f.write("="*70 + "\n\n")
        f.write(classification_report(y_test, y_pred_test, target_names=label_encoder.classes_))
    mlflow.log_artifact(report_path)
    
    mlflow.end_run()
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"\nFinal Test Accuracy: {test_results['accuracy']:.2%}")
    
    return clf, scaler, feature_names, label_encoder, test_results

# ============================================================================
# CLI
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train hybrid model (BERT + RandomForest) on EBBU dataset")
    parser.add_argument('--data_path', type=str, default='data/url_datasets/EBBU', 
                       help='Path to EBBU dataset')
    parser.add_argument('--pretrained_path', type=str, default='amahdaouy/DomURLs_BERT', 
                       help='Pretrained transformer model path')
    parser.add_argument('--max_length', type=int, default=128, 
                       help='Max sequence length for BERT')
    parser.add_argument('--batch_size', type=int, default=32, 
                       help='Batch size for embedding extraction')
    parser.add_argument('--n_estimators', type=int, default=100, 
                       help='Number of trees in Random Forest')
    parser.add_argument('--max_depth', type=int, default=20, 
                       help='Max depth of Random Forest trees')
    parser.add_argument('--sample_size', type=int, default=0, 
                       help='Sample size for testing (0 = full dataset)')
    parser.add_argument('--seed', type=int, default=3407, 
                       help='Random seed')
    
    args = parser.parse_args()
    
    main(args=args)
