"""Inference module for predictions on new text."""

import warnings

# Suppress transformers library deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*__path__.*")
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
from pathlib import Path
import json

from config import DEVICE, MODEL_NAME, NUM_CLASSES, MAX_LEN, XLEX_PATH
from models.model import EnhancedFinSentiBERT
from utils.tokenizer import tokenize
from xlex.xlex import XLex


class FinancialSentimentClassifier:
    """Wrapper class for sentiment classification inference."""
    
    LABEL_MAP = {0: "positive", 1: "negative", 2: "neutral"}
    
    def __init__(self, model_path="models_checkpoint/best_model.pt", xlex_path=XLEX_PATH):
        """Initialize classifier with model and XLex.
        
        Args:
            model_path: Path to trained model checkpoint
            xlex_path: Path to XLex JSON file
        """
        self.device = DEVICE
        self.model = EnhancedFinSentiBERT().to(self.device)
        self.xlex = XLex(xlex_path)
        
        # Load model weights
        if Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"✓ Model loaded from: {model_path}")
        else:
            print(f"Warning: Model checkpoint not found at {model_path}")
        
        self.model.eval()
    
    def predict(self, text, return_probabilities=False):
        """Predict sentiment for text.
        
        Args:
            text: Input text string
            return_probabilities: If True, return probability scores
        
        Returns:
            dict: Prediction results with label and confidence
        """
        with torch.no_grad():
            input_ids, attention_mask, tokens = tokenize(text)
            lex = self.xlex.encode(tokens)
            
            # Add batch dimension
            input_ids = input_ids.unsqueeze(0).to(self.device)
            attention_mask = attention_mask.unsqueeze(0).to(self.device)
            lex = lex.unsqueeze(0).to(self.device)
            
            # Get predictions
            logits = self.model(input_ids, attention_mask, lex)
            probs = torch.softmax(logits, dim=-1)[0]
            pred_label = torch.argmax(probs, dim=-1).item()
            confidence = probs[pred_label].item()
        
        result = {
            "text": text,
            "label": self.LABEL_MAP[pred_label],
            "confidence": confidence,
            "label_id": pred_label
        }
        
        if return_probabilities:
            result["probabilities"] = {
                self.LABEL_MAP[i]: probs[i].item() 
                for i in range(NUM_CLASSES)
            }
        
        return result
    
    def predict_batch(self, texts, return_probabilities=False):
        """Predict sentiment for multiple texts.
        
        Args:
            texts: List of text strings
            return_probabilities: If True, return probability scores
        
        Returns:
            list: List of prediction results
        """
        results = []
        for text in texts:
            result = self.predict(text, return_probabilities=return_probabilities)
            results.append(result)
        return results


def main():
    """Example inference usage."""
    print("\n" + "="*60)
    print("Financial Sentiment Classification - Inference")
    print("="*60 + "\n")
    
    # Initialize classifier
    classifier = FinancialSentimentClassifier()
    
    # Test examples
    test_texts = [
        "The company reported strong earnings growth this quarter.",
        "Stock prices fell significantly due to market concerns.",
        "The results were within expected ranges.",
    ]
    
    print("Testing predictions:\n")
    for text in test_texts:
        result = classifier.predict(text, return_probabilities=True)
        print(f"Text: {result['text']}")
        print(f"Prediction: {result['label'].upper()} (confidence: {result['confidence']:.2%})")
        print(f"Probabilities: {result['probabilities']}")
        print("-" * 60)

if __name__ == "__main__":
    main()
