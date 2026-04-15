# 🎯 Financial Sentiment Analysis - Frontend Quick Start

This directory contains **4 fully-functional frontends** for analyzing the sentiment of financial news and articles using a trained BERT model enhanced with financial lexicon (XLex).

## 🚀 Quick Start (30 seconds)

```bash
# 1. Install dependencies (first time only)
pip install -r requirements.txt

# 2. Run the smart launcher
python launch.py

# 3. Choose your interface and start analyzing!
```

---

## 📊 Available Frontends

### 1. **Streamlit Web App** (Recommended for most users)
**Best for:** Interactive analysis, batch processing, visualizations

```bash
streamlit run app.py
```

✨ Features:
- Beautiful interactive web UI  
- Single & batch text analysis
- Interactive probability charts
- Settings sidebar
- **Access:** http://localhost:8501

---

### 2. **Flask REST API** (Best for developers)
**Best for:** Integrating into other applications, building custom frontends

```bash
python api.py
```

✨ Features:
- RESTful endpoints
- JSON input/output
- CORS enabled for web integration
- Batch prediction support
- **Access:** http://localhost:5000

**Example API calls:**
```bash
# Single prediction
curl -X POST http://localhost:5000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Strong earnings beat!"}'

# Batch prediction
curl -X POST http://localhost:5000/api/v1/predict-batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Text 1", "Text 2"]}'
```

---

### 3. **HTML Web Frontend** (Lightweight, no Streamlit needed)
**Best for:** Lightweight deployment, running locally

```bash
# Start API server (in one terminal)
python api.py

# Then open index.html in browser
# Option A: Open directly
open index.html  # macOS
xdg-open index.html  # Linux
start index.html  # Windows

# Option B: Use Python HTTP server
python -m http.server 8000
# Visit: http://localhost:8000/index.html
```

✨ Features:
- Professional, responsive design
- Single & batch analysis tabs
- Real-time charts and statistics
- Mobile-friendly
- **No Streamlit dependency**

---

### 4. **Command-Line Tool** (Best for automation)
**Best for:** Batch processing, scripting, pipeline integration

```bash
# Single text
python cli.py "The company exceeded revenue expectations!"

# Interactive mode
python cli.py

# Batch from file
python cli.py --file articles.txt

# JSON output
python cli.py "Some text" --json

# With probabilities
python cli.py "Some text" --verbose
```

✨ Features:
- Color-coded output
- Multiple input modes
- JSON support
- Batch processing
- No browser needed

---

## 🎬 Run Everything at Once

```bash
python launch.py
# Select option 5: Run All
```

This will:
1. ✅ Check model & dependencies
2. ✅ Start Flask API on port 5000
3. ✅ Open HTML frontend in browser
4. ✅ Keep everything running together

---

## 📝 What You Can Do With These Frontends

### Analyze News Articles
```
Paste financial news → Get sentiment classification
```

### Monitor Markets
```
Input multiple stock updates → See sentiment distribution
```

### Batch Process Files
```
articles.txt (1000 lines) → Bulk sentiment analysis → Export results
```

### Integrate Into Apps
```
Use Flask API → Build your own UI → Export JSON
```

### Automate Workflows
```
CLI + shell scripts → Process feeds automatically
```

---

## 📖 Full Documentation

For detailed information, see **[FRONTEND_GUIDE.md](FRONTEND_GUIDE.md)**

Topics covered:
- Step-by-step setup
- Detailed usage examples for each frontend
- API endpoint documentation
- Code examples (Python, JavaScript)
- Troubleshooting
- Performance tips
- Deployment guides

---

## 🛠️ Requirements & Setup

### System Requirements
- Python 3.8+
- 4GB RAM (8GB recommended for GPU)
- CUDA optional (for GPU acceleration)

### Installation
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Verify Installation
```bash
# Check model exists
ls models_checkpoint/best_model.pt

# Test CLI
python cli.py "Test text"

# Should output: ✓ Model loaded + sentiment result
```

---

## 🎨 Frontend Comparison

| Feature | Streamlit | Flask API | HTML | CLI |
|---------|-----------|-----------|------|-----|
| Web UI | ✅ | ✅* | ✅ | ❌ |
| Batch Analysis | ✅ | ✅ | ✅ | ✅ |
| Charts/Viz | ✅ | ❌ | ✅ | ✅ |
| JSON Output | ❌ | ✅ | ✅ | ✅ |
| Automation | ❌ | ✅ | ❌ | ✅ |
| Mobile Friendly | ⚠️ | ❌ | ✅ | ❌ |
| Installation | Streamlit | Flask | None | None |
| Learning Curve | Easy | Medium | Easy | Easy |

\* API provides JSON for frontend integration

---

## 🎯 Common Use Cases & Recommendations

### "I want a simple web interface"
→ **Use Streamlit**: `streamlit run app.py`

### "I'm building a web app"
→ **Use Flask API**: `python api.py`

### "I need lightweight deployment"
→ **Use HTML frontend**: Works with any API

### "I'm processing thousands of articles"
→ **Use CLI**: `python cli.py --file articles.txt`

### "I want to integrate into Python code"
→ **Use API**: Call endpoints via requests library

### "I'm not sure what I want"
→ **Use Launcher**: `python launch.py`

---

## 📊 Example Outputs

### Streamlit Web App
```
Sentiment: 🟢 POSITIVE
Confidence: 94.2%
[Chart showing: Positive 94%, Negative 4%, Neutral 2%]
```

### CLI Tool
```
============================================================
FINANCIAL SENTIMENT ANALYSIS
============================================================
📝 Text: "Strong earnings beat market expectations!"
🎯 Sentiment: 🟢 POSITIVE (94.2%)
============================================================
```

### API Response
```json
{
  "success": true,
  "result": {
    "text": "Strong earnings beat market expectations!",
    "label": "positive",
    "confidence": 0.942,
    "probabilities": {
      "positive": 0.942,
      "negative": 0.038,
      "neutral": 0.020
    }
  }
}
```

---

## ⚡ Performance Tips

1. **GPU Acceleration**: Install CUDA for 3-5x speedup
2. **Batch Processing**: Process multiple texts efficiently
3. **Caching**: Store results for frequent queries
4. **API Reuse**: Use same API instance across requests

---

## 🆘 Troubleshooting

### Model not found
```
❌ Error: "Model checkpoint not found"
✅ Solution: Run training first
   python main.py
```

### Streamlit not installed
```
❌ Error: "No module named 'streamlit'"
✅ Solution: Install dependencies
   pip install -r requirements.txt
```

### Port already in use
```
❌ Error: "Address already in use"
✅ Solution: Kill existing process
   lsof -i :5000  # Find it
   kill -9 <PID>  # Kill it
```

---

## 🔗 API Quick Reference

```
GET    /health                    # Health check
GET    /api/v1/models             # Model info
POST   /api/v1/predict            # Single prediction
POST   /api/v1/predict-batch      # Batch predictions
```

More details: See [FRONTEND_GUIDE.md](FRONTEND_GUIDE.md) → "REST API" section

---

## 📞 Getting Help

1. **Check FRONTEND_GUIDE.md** for detailed docs
2. **Review error messages** - they're descriptive
3. **Check code comments** - all files are well-documented
4. **Try simpler interface** - CLI for debugging

---

## 📦 What's Included

```
✅ 4 fully functional frontends
✅ Complete documentation
✅ Example code for integration
✅ Smart launcher utility
✅ All dependencies listed
✅ Troubleshooting guide
✅ Performance optimization tips
```

---

## 🚀 Ready to Start?

```bash
# Option 1: Smart choice
python launch.py

# Option 2: Streamlit UI
streamlit run app.py

# Option 3: API backend
python api.py

# Option 4: CLI tool
python cli.py

# Option 5: See docs
cat FRONTEND_GUIDE.md
```

---

**Happy analyzing! 📈**

Questions? Check [FRONTEND_GUIDE.md](FRONTEND_GUIDE.md) for comprehensive documentation.
