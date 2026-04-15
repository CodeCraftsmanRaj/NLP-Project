"""Flask REST API for Financial Sentiment Analysis."""

from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from inference import FinancialSentimentClassifier
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load classifier
try:
    classifier = FinancialSentimentClassifier()
    logger.info("✓ Financial Sentiment Classifier loaded successfully")
except Exception as e:
    logger.error(f"✗ Failed to load classifier: {e}")
    classifier = None


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "classifier_loaded": classifier is not None
    }), 200


@app.route('/api/v1/predict', methods=['POST'])
def predict_single():
    """
    Predict sentiment for single text.
    
    Request JSON:
    {
        "text": "Your financial text here",
        "return_probabilities": true (optional)
    }
    
    Returns: Prediction with label, confidence, and optional probabilities
    """
    if not classifier:
        return jsonify({"error": "Classifier not loaded"}), 500
    
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({"error": "Missing 'text' in request body"}), 400
        
        text = data['text'].strip()
        if not text:
            return jsonify({"error": "Text cannot be empty"}), 400
        
        return_probs = data.get('return_probabilities', False)
        
        # Predict
        result = classifier.predict(text, return_probabilities=return_probs)
        
        return jsonify({
            "success": True,
            "result": result
        }), 200
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/v1/predict-batch', methods=['POST'])
def predict_batch():
    """
    Predict sentiment for multiple texts.
    
    Request JSON:
    {
        "texts": ["Text 1", "Text 2", "Text 3"],
        "return_probabilities": true (optional)
    }
    
    Returns: List of predictions with summary statistics
    """
    if not classifier:
        return jsonify({"error": "Classifier not loaded"}), 500
    
    try:
        data = request.get_json()
        
        if not data or 'texts' not in data:
            return jsonify({"error": "Missing 'texts' in request body"}), 400
        
        texts = data['texts']
        if not isinstance(texts, list):
            return jsonify({"error": "'texts' must be a list"}), 400
        
        if not texts:
            return jsonify({"error": "Texts list cannot be empty"}), 400
        
        # Filter and validate texts
        texts = [t.strip() for t in texts if isinstance(t, str) and t.strip()]
        if not texts:
            return jsonify({"error": "No valid texts to analyze"}), 400
        
        return_probs = data.get('return_probabilities', False)
        
        # Predict batch
        results = classifier.predict_batch(texts, return_probabilities=return_probs)
        
        # Calculate summary statistics
        sentiments = [r['label'] for r in results]
        positive_count = sentiments.count('positive')
        negative_count = sentiments.count('negative')
        neutral_count = sentiments.count('neutral')
        avg_confidence = sum(r['confidence'] for r in results) / len(results)
        
        summary = {
            "total_texts": len(results),
            "positive": positive_count,
            "negative": negative_count,
            "neutral": neutral_count,
            "average_confidence": avg_confidence,
            "positive_percentage": (positive_count / len(results)) * 100,
            "negative_percentage": (negative_count / len(results)) * 100,
            "neutral_percentage": (neutral_count / len(results)) * 100
        }
        
        return jsonify({
            "success": True,
            "results": results,
            "summary": summary
        }), 200
    
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/v1/models', methods=['GET'])
def get_models_info():
    """Get information about available models."""
    return jsonify({
        "name": "EnhancedFinSentiBERT",
        "description": "BERT-based financial sentiment classifier enhanced with XLex",
        "classes": ["positive", "negative", "neutral"],
        "language": "English",
        "domain": "Financial News and Documents",
        "features": {
            "bert_model": "bert-base-uncased",
            "xlex_dimensions": 6,
            "max_sequence_length": 128
        }
    }), 200


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({"error": "Internal server error"}), 500


if __name__ == '__main__':
    # Get port from environment or use 5000
    port = int(os.environ.get('PORT', 5000))
    
    print("\n" + "="*60)
    print("Financial Sentiment Analysis API")
    print("="*60)
    print(f"\n🚀 Starting server on http://localhost:{port}")
    print("\nEndpoints:")
    print("  GET  /health - Health check")
    print("  GET  /api/v1/models - Model information")
    print("  POST /api/v1/predict - Single text prediction")
    print("  POST /api/v1/predict-batch - Batch predictions")
    print("\n" + "="*60 + "\n")
    
    app.run(host='0.0.0.0', port=port, debug=False)
