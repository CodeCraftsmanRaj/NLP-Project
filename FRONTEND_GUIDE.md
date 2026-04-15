"""Frontend Setup and Usage Guide for Financial Sentiment Analysis"""

# Financial Sentiment Analysis Frontend Guide

This document explains how to use the three different frontends created for the Financial Sentiment Analysis project.

## 📋 Quick Overview

| Frontend | Use Case | Command |
|----------|----------|---------|
| **Streamlit Web App** | Interactive UI, batch analysis | `streamlit run app.py` |
| **CLI Tool** | Quick command-line predictions | `python cli.py "text"` |
| **Flask REST API** | Backend service, integration | `python api.py` |

---

## 🔧 Installation

### Prerequisites
- Python 3.8+
- CUDA (optional, for GPU acceleration)

### Install Dependencies

Create a virtual environment:
```bash
cd /home/arsalan/Desktop/NLP-Project
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install required packages:
```bash
pip install --upgrade pip
pip install torch transformers
pip install streamlit plotly
pip install flask flask-cors
pip install colorama tabulate
pip install requests  # For testing API
```

Or install all at once from requirements:
```bash
pip install -r requirements.txt
```

---

## 🎯 Interface 1: Streamlit Web App (Recommended for Users)

### Launch
```bash
streamlit run app.py
```

Your browser should open to `http://localhost:8501`

### Features
- ✅ **Single Text Analysis Tab**
  - Paste financial news articles
  - Instant sentiment classification (Positive/Negative/Neutral)
  - Confidence scores
  - Probability distribution visualization
  - Example text provided

- ✅ **Batch Analysis Tab**
  - Analyze multiple texts at once
  - Results displayed in table format
  - Sentiment distribution pie chart
  - Summary statistics

- ✅ **Sidebar Settings**
  - Toggle probability display
  - Toggle JSON output
  - Model information

### Usage Example
1. Open the app (URL shown after running the command)
2. Paste this example text:
```
"Apple reported record Q3 revenues, with iPhone sales surging 25% year-over-year. 
Supply chain improvements and strong Asian demand boosted profitability."
```
3. Click "Analyze" (or just start typing to auto-analyze)
4. View results with confidence scores and probability distribution

---

## ⌨️ Interface 2: Command-Line Tool (Best for Automation)

### Single Text Analysis
```bash
python cli.py "The company faces significant financial challenges ahead."
```

Output:
```
============================================================
FINANCIAL SENTIMENT ANALYSIS
============================================================

📝 Text: The company faces significant financial challenges ahead.

🎯 Sentiment: 📉 NEGATIVE (92.5%)

============================================================
```

### With Probability Distribution
```bash
python cli.py "Strong earnings beat market expectations!" --verbose
```

### Batch Analysis from File
```bash
# Create a file with texts (one per line)
echo "Strong quarterly results
Declining market share  
Revenue in line with forecasts" > articles.txt

# Analyze
python cli.py --file articles.txt
```

### JSON Output (for scripting/integration)
```bash
python cli.py "Some financial text" --json
```

Output:
```json
{
  "text": "Some financial text",
  "label": "neutral",
  "confidence": 0.65,
  "label_id": 2,
  "probabilities": {
    "positive": 0.2,
    "negative": 0.15,
    "neutral": 0.65
  }
}
```

### Interactive Mode
Simply run without arguments:
```bash
python cli.py
# Then enter text interactively, type 'quit' to exit
```

### Full CLI Options
```bash
python cli.py --help

Options:
  -f, --file TEXT           Analyze texts from file
  -v, --verbose            Show detailed probabilities
  -j, --json               Output as JSON
  --no-summary             Skip summary statistics
  -m, --model PATH         Custom model checkpoint
```

---

## 🔌 Interface 3: Flask REST API (For Developers/Integration)

### Start API Server
```bash
python api.py
```

Server runs on `http://localhost:5000`

### API Endpoints

#### 1. Health Check
```bash
curl http://localhost:5000/health
```

#### 2. Single Text Prediction
```bash
curl -X POST http://localhost:5000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Company exceeded revenue expectations!"}'
```

Response:
```json
{
  "success": true,
  "result": {
    "text": "Company exceeded revenue expectations!",
    "label": "positive",
    "confidence": 0.94,
    "label_id": 0,
    "probabilities": {
      "positive": 0.94,
      "negative": 0.04,
      "neutral": 0.02
    }
  }
}
```

#### 3. Batch Predictions
```bash
curl -X POST http://localhost:5000/api/v1/predict-batch \
  -H "Content-Type: application/json" \
  -d '{"texts": [
    "Strong earnings beat",
    "Declining market share",
    "Revenue in line with forecasts"
  ]}'
```

Response:
```json
{
  "success": true,
  "results": [...],
  "summary": {
    "total_texts": 3,
    "positive": 1,
    "negative": 1,
    "neutral": 1,
    "average_confidence": 0.87,
    "positive_percentage": 33.33,
    "negative_percentage": 33.33,
    "neutral_percentage": 33.33
  }
}
```

#### 4. Model Information
```bash
curl http://localhost:5000/api/v1/models
```

### Python Client Example
```python
import requests
import json

API_URL = "http://localhost:5000/api/v1"

# Single prediction
response = requests.post(f"{API_URL}/predict", json={
    "text": "Excellent quarterly performance!",
    "return_probabilities": True
})
print(json.dumps(response.json(), indent=2))

# Batch predictions
response = requests.post(f"{API_URL}/predict-batch", json={
    "texts": ["Text 1", "Text 2"],
    "return_probabilities": True
})
print(json.dumps(response.json(), indent=2))
```

### JavaScript/Fetch Example
```javascript
const API_URL = "http://localhost:5000/api/v1";

async function analyzeSentiment(text) {
  const response = await fetch(`${API_URL}/predict`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      text: text,
      return_probabilities: true
    })
  });
  
  const data = await response.json();
  return data.result;
}

// Usage
analyzeSentiment("Strong earnings report!").then(result => {
  console.log(result);
  // Output: { text: "...", label: "positive", confidence: 0.94, ... }
});
```

---

## 📊 Output Formats

### Sentiment Labels
- **`positive`** 🟢 - Bullish, optimistic financial news
- **`negative`** 🔴 - Bearish, pessimistic financial news
- **`neutral`** 🟡 - No clear sentiment or mixed signals

### Confidence Score
- Range: 0.0 - 1.0 (or 0% - 100%)
- Higher = more confident in prediction
- Above 0.8 is typically very reliable

---

## 🎒 Use Cases

### 1. News Sentiment Analysis
```bash
# Analyze latest financial news
curl https://newsapi.org/v2/everything?q=tech OR company | \
  jq '.articles[].description' | \
  python cli.py --file /dev/stdin
```

### 2. Real-time Dashboard
Use the **Streamlit app** for live visualization and monitoring of sentiment trends.

### 3. Automated Alerts
```python
# Send alerts for negative financial news
import requests

articles = fetch_articles()  # Your data source
for article in articles:
    result = requests.post("http://localhost:5000/api/v1/predict", 
                          json={"text": article["content"]})
    if result.json()["result"]["label"] == "negative":
        send_alert(f"⚠️ Negative news: {article['title']}")
```

### 4. Batch Reporting
```bash
# Analyze weekly earnings reports
python cli.py --file earnings_reports.txt --no-summary > report.txt
```

---

## 🐛 Troubleshooting

### Model not loading
**Error:** "Warning: Model checkpoint not found"
**Solution:** Run training first or check `models_checkpoint/best_model.pt` exists
```bash
python main.py  # Train the model
```

### GPU issues
**Error:** CUDA out of memory
**Solution:** Use CPU or reduce batch size
```python
# In config.py, set DEVICE = "cpu"
```

### Port already in use
**Error:** "Address already in use"
**Solution:** Kill existing process or use different port
```bash
# Find and kill process
lsof -i :5000  # on Mac/Linux
netstat -ano | findstr :5000  # on Windows

# Or use different port
python api.py  # Default 5000
PORT=8000 python api.py  # Use 8000
```

---

## 📈 Performance Tips

1. **Use CLI for high-volume analysis** - Faster than API overhead
2. **Batch predictions** - Process multiple texts efficiently
3. **Cache results** - Store predictions to avoid re-analyzing
4. **Enable GPU** - Install CUDA for 3-5x speedup

---

## 📚 Model Information

- **Base Model:** BERT (bert-base-uncased)
- **Enhancement:** Financial Lexicon (XLex) - 6 dimensions
- **Architecture:** Multi-head attention fusion
- **Classes:** 3 (Positive, Negative, Neutral)
- **Training Data:** FinancialPhraseBank
- **Max Input Length:** 128 tokens (~512 characters)

---

## ✅ Quick Start Checklist

- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Verify model checkpoint: `ls models_checkpoint/best_model.pt`
- [ ] Test CLI: `python cli.py "test text"`
- [ ] Test Streamlit: `streamlit run app.py`
- [ ] Test API: `python api.py` then `curl http://localhost:5000/health`
- [ ] Try batch analysis with sample file

---

## 🎓 Next Steps

1. **Customize the UI** - Edit `app.py` for your branding
2. **Add authentication** - Secure the API with API keys
3. **Deploy to cloud** - Use Heroku, AWS, or Azure
4. **Add more features** - Text preprocessing, visualization, export
5. **Fine-tune model** - Train on your own financial data

---

Generated: April 2026
