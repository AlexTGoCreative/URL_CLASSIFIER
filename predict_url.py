"""
EBBU Hybrid Model - URL Phishing Detection Inference Script
Load saved hybrid model and predict on hardcoded URLs
"""

import pickle
import numpy as np
from pathlib import Path
from collections import Counter
from urllib.parse import urlparse
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# HANDCRAFTED FEATURE EXTRACTOR (same as in main_ebbu_hybrid.py)
# ============================================================================

class HandcraftedFeatureExtractor:
    """Extract handcrafted features to complement BERT embeddings"""
    
    def __init__(self, alexa_path='data/url_datasets/EBBU/top-1m.csv'):
        self.alexa_domains = None
        self.alexa_path = alexa_path
    
    def extract_features(self, url):
        """Extract handcrafted features from URL"""
        features = {}
        
        try:
            parsed = urlparse(url)
            domain = parsed.netloc
            path = parsed.path
            query = parsed.query
            
            # Domain features
            registered_domain = self._get_registered_domain(domain)
            features['domain_length'] = len(domain)
            features['domain_randomness'] = self._calculate_domain_randomness(registered_domain)
            features['is_random_domain'] = int(features['domain_randomness'] > 0.7)
            features['has_subdomain'] = int('.' in domain and domain.count('.') > 1)
            features['subdomain_count'] = domain.count('.') - 1 if '.' in domain else 0
            
            # Path features
            features['path_length'] = len(path)
            features['path_depth'] = path.count('/')
            features['has_query'] = int(len(query) > 0)
            features['query_length'] = len(query)
            
            # URL character features
            features['url_length'] = len(url)
            features['digit_count'] = sum(c.isdigit() for c in url)
            features['special_char_count'] = sum(not c.isalnum() and c not in ['.', '/', ':', '-', '_'] for c in url)
            features['has_ip'] = int(self._has_ip_address(domain))
            features['https'] = int(parsed.scheme == 'https')
            
            # Alexa ranking features
            features['alexa_top_1m'] = int(self._is_alexa_domain(domain))
            features['alexa_top_100k'] = int(self._is_alexa_domain(domain, top_n=100000))
            
        except Exception as e:
            # Return default features on error
            features = {
                'domain_length': 0, 'domain_randomness': 0.0, 'is_random_domain': 0,
                'has_subdomain': 0, 'subdomain_count': 0, 'path_length': 0,
                'path_depth': 0, 'has_query': 0, 'query_length': 0,
                'url_length': 0, 'digit_count': 0, 'special_char_count': 0,
                'has_ip': 0, 'https': 0, 'alexa_top_1m': 0, 'alexa_top_100k': 0
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
    
    def _load_alexa_domains(self):
        """Load Alexa Top 1M domains"""
        if self.alexa_domains is not None:
            return
        try:
            df = pd.read_csv(self.alexa_path, names=['rank', 'domain'])
            self.alexa_domains = set(df['domain'].str.lower())
            print(f"✓ Loaded {len(self.alexa_domains)} Alexa domains")
        except Exception as e:
            print(f"⚠ Warning: Could not load Alexa domains: {e}")
            self.alexa_domains = set()
    
    def _is_alexa_domain(self, domain, top_n=1000000):
        """Check if domain is in Alexa Top N list"""
        if self.alexa_domains is None:
            self._load_alexa_domains()
        registered_domain = self._get_registered_domain(domain)
        return registered_domain.lower() in self.alexa_domains

# ============================================================================
# BERT EMBEDDER (same as in main_ebbu_hybrid.py)
# ============================================================================

import torch
from transformers import AutoTokenizer, AutoModel

class DomURLsBERTEmbedder:
    """Extract embeddings using DomURLs_BERT"""
    
    def __init__(self, model_name='amahdaouy/DomURLs_BERT', max_length=128):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"✓ Using device: {self.device}")
        
        print(f"✓ Loading {model_name} model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.max_length = max_length
        print("✓ Model loaded successfully!")
    
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
            print(f"✗ Error getting embedding for {url}: {e}")
            return np.zeros(768)

# ============================================================================
# PHISHING DETECTOR CLASS
# ============================================================================

class PhishingDetector:
    """
    Phishing URL Detector using Hybrid Model (BERT + Random Forest)
    """
    
    def __init__(self, model_path, bert_model='amahdaouy/DomURLs_BERT'):
        """
        Initialize detector with saved model
        
        Args:
            model_path: Path to saved model pickle file
            bert_model: BERT model name/path for embeddings
        """
        print("\n" + "="*70)
        print("INITIALIZING PHISHING DETECTOR")
        print("="*70)
        
        # Load saved model
        print(f"\n✓ Loading model from: {model_path}")
        with open(model_path, 'rb') as f:
            model_package = pickle.load(f)
        
        self.classifier = model_package['classifier']
        self.scaler = model_package['scaler']
        self.feature_names = model_package['feature_names']
        self.label_encoder = model_package['label_encoder']
        self.model_type = model_package.get('model_type', 'Unknown')
        
        print(f"✓ Model type: {self.model_type}")
        print(f"✓ Features: {len(self.feature_names)}")
        print(f"✓ Classes: {list(self.label_encoder.classes_)}")
        
        # Initialize feature extractors
        print("\n✓ Initializing feature extractors...")
        self.embedder = DomURLsBERTEmbedder(model_name=bert_model)
        self.feature_extractor = HandcraftedFeatureExtractor()
        
        print("\n" + "="*70)
        print("DETECTOR READY!")
        print("="*70 + "\n")
    
    def predict(self, url, verbose=True):
        """
        Predict if URL is phishing or legitimate
        
        Args:
            url: URL string to classify
            verbose: Print detailed analysis
        
        Returns:
            dict with prediction results
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"ANALYZING URL: {url}")
            print(f"{'='*70}")
        
        # Extract features
        if verbose:
            print("\n[1] Extracting handcrafted features...")
        handcrafted = self.feature_extractor.extract_features(url)
        handcrafted_array = np.array([list(handcrafted.values())])
        
        if verbose:
            print(f"    • Domain length: {handcrafted['domain_length']}")
            print(f"    • Domain randomness: {handcrafted['domain_randomness']:.2f}")
            print(f"    • URL length: {handcrafted['url_length']}")
            print(f"    • HTTPS: {'Yes' if handcrafted['https'] else 'No'}")
            print(f"    • In Alexa Top 1M: {'Yes' if handcrafted['alexa_top_1m'] else 'No'}")
        
        if verbose:
            print("\n[2] Extracting BERT embeddings...")
        bert_embedding = self.embedder.get_embedding(url)
        bert_array = bert_embedding.reshape(1, -1)
        
        if verbose:
            print(f"    • Embedding dimension: {bert_array.shape[1]}")
        
        # Combine features
        if verbose:
            print("\n[3] Combining features...")
        X = np.hstack([handcrafted_array, bert_array])
        
        # Scale and predict
        if verbose:
            print("\n[4] Running prediction...")
        X_scaled = self.scaler.transform(X)
        prediction = self.classifier.predict(X_scaled)[0]
        probabilities = self.classifier.predict_proba(X_scaled)[0]
        
        # Get label
        predicted_label = self.label_encoder.inverse_transform([prediction])[0]
        
        # Build result
        result = {
            'url': url,
            'prediction': predicted_label,
            'prediction_numeric': int(prediction),
            'probabilities': {
                label: float(prob) 
                for label, prob in zip(self.label_encoder.classes_, probabilities)
            },
            'confidence': float(max(probabilities)),
            'is_phishing': predicted_label == 'phish',
            'handcrafted_features': handcrafted
        }
        
        if verbose:
            self._print_verdict(result)
        
        return result
    
    def _print_verdict(self, result):
        """Print formatted verdict"""
        print("\n" + "="*70)
        print("VERDICT")
        print("="*70)
        
        # Prediction
        if result['is_phishing']:
            verdict_symbol = "🚨 PHISHING"
            verdict_color = "DANGER"
        else:
            verdict_symbol = "✅ LEGITIMATE"
            verdict_color = "SAFE"
        
        print(f"\n{verdict_symbol}")
        print(f"Confidence: {result['confidence']*100:.2f}%")
        
        # Probabilities
        print(f"\nClass Probabilities:")
        for label, prob in result['probabilities'].items():
            bar_length = int(prob * 50)
            bar = "█" * bar_length + "░" * (50 - bar_length)
            print(f"  {label:>10s}: {bar} {prob*100:5.2f}%")
        
        # Key features
        features = result['handcrafted_features']
        print(f"\nKey Indicators:")
        print(f"  • Domain Randomness: {features['domain_randomness']:.2f} {'⚠ HIGH' if features['is_random_domain'] else '✓ Normal'}")
        print(f"  • URL Length: {features['url_length']} chars")
        print(f"  • HTTPS: {'✓ Yes' if features['https'] else '✗ No'}")
        print(f"  • Alexa Ranked: {'✓ Yes' if features['alexa_top_1m'] else '✗ No'}")
        print(f"  • Has IP in domain: {'⚠ Yes' if features['has_ip'] else '✓ No'}")
        
        print("="*70 + "\n")
    
    def predict_batch(self, urls):
        """Predict multiple URLs"""
        results = []
        for url in urls:
            result = self.predict(url, verbose=False)
            results.append(result)
        return results

# ============================================================================
# MAIN EXECUTION - HARDCODED TEST URLS
# ============================================================================

def main():
    """
    Main inference script with hardcoded test URLs
    """
    
    # ========== CONFIGURATION ==========
    # Path to saved model (update this to your actual model path)
    MODEL_PATH = "mlruns/ckpts/ebbu_hybrid_<run_id>/hybrid_model.pkl"
    
    # If you want to use the latest model automatically:
    import os
    import glob
    
    # Find most recent model
    model_files = glob.glob("mlruns/ckpts/ebbu_hybrid_*/hybrid_model.pkl")
    if model_files:
        MODEL_PATH = max(model_files, key=os.path.getctime)
        print(f"Using most recent model: {MODEL_PATH}")
    else:
        print("⚠ No saved models found. Please run main_ebbu_hybrid.py first!")
        print("Expected path: mlruns/ckpts/ebbu_hybrid_*/hybrid_model.pkl")
        return
    
    # ========== HARDCODED TEST URLS ==========
    test_urls = [
        # Legitimate URLs
        "https://www.google.com",
        "https://www.github.com/pytorch/pytorch",
        "https://stackoverflow.com/questions/tagged/python",
        "https://www.amazon.com/products",
        
        # Suspicious/Phishing-like URLs
        "http://paypal-verify-account.tk/login.php",
        "http://192.168.1.1/admin/login.html",
        "http://apple-id-verify.ml/secure/update.php",
        "http://www.bankoamerica-secure.xyz/signin",
        "https://docs.google.com.phishing-site.ru/login",
    ]
    
    # ========== INITIALIZE DETECTOR ==========
    detector = PhishingDetector(
        model_path=MODEL_PATH,
        bert_model='amahdaouy/DomURLs_BERT'
    )
    
    # ========== RUN PREDICTIONS ==========
    print("\n" + "="*70)
    print("RUNNING PREDICTIONS ON HARDCODED URLs")
    print("="*70)
    
    results = []
    for i, url in enumerate(test_urls, 1):
        print(f"\n[{i}/{len(test_urls)}] Testing URL...")
        result = detector.predict(url, verbose=True)
        results.append(result)
    
    # ========== SUMMARY ==========
    print("\n" + "="*70)
    print("SUMMARY OF ALL PREDICTIONS")
    print("="*70 + "\n")
    
    phishing_count = sum(1 for r in results if r['is_phishing'])
    legitimate_count = len(results) - phishing_count
    
    print(f"Total URLs analyzed: {len(results)}")
    print(f"  • Legitimate: {legitimate_count}")
    print(f"  • Phishing: {phishing_count}")
    
    print("\nDetailed Results:")
    print("-" * 70)
    for i, result in enumerate(results, 1):
        status = "🚨 PHISHING" if result['is_phishing'] else "✅ LEGIT"
        conf = result['confidence'] * 100
        url_short = result['url'][:60] + "..." if len(result['url']) > 60 else result['url']
        print(f"{i:2d}. {status} ({conf:5.1f}%) | {url_short}")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
