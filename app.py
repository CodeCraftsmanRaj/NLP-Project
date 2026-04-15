"""Streamlit web app for Financial Sentiment Analysis."""

import streamlit as st
import plotly.graph_objects as go
from inference import FinancialSentimentClassifier
import time

# Page config
st.set_page_config(
    page_title="Financial Sentiment Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .sentiment-positive {
        color: #09AB3B;
        font-weight: bold;
    }
    .sentiment-negative {
        color: #FF2B2B;
        font-weight: bold;
    }
    .sentiment-neutral {
        color: #FF9500;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize classifier in session state
@st.cache_resource
def load_classifier():
    """Load the financial sentiment classifier."""
    with st.spinner("Loading model..."):
        classifier = FinancialSentimentClassifier()
    return classifier

# Title and header
st.title("📈 Financial Sentiment Analysis")
st.markdown("""
    Analyze the sentiment of financial news articles and documents.  
    This AI-powered tool uses BERT and financial lexicon features to classify text as **Positive**, **Negative**, or **Neutral**.
""")

# Sidebar
with st.sidebar:
    st.header("About")
    st.markdown("""
    ### How it works:
    1. **Input** your news article or financial text
    2. **BERT Model** analyzes the language
    3. **Financial Lexicon (XLex)** adds domain knowledge
    4. **Classification** predicts sentiment with confidence
    
    ### Model Details:
    - Dataset: FinancialPhraseBank
    - Architecture: EnhancedFinSentiBERT
    - Features: BERT + XLex (6-dim)
    """)
    
    st.divider()
    
    st.header("Settings")
    show_probabilities = st.checkbox("Show probability distribution", value=True)
    show_raw_output = st.checkbox("Show raw JSON output", value=False)

# Load classifier
classifier = load_classifier()

# Main content
tab1, tab2 = st.tabs(["📝 Single Text Analysis", "📊 Batch Analysis"])

# Tab 1: Single Text Analysis
with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Input Text")
        input_method = st.radio("Choose input method:", ["Paste Text", "Example"], horizontal=True)
        
        if input_method == "Paste Text":
            user_text = st.text_area(
                "Paste your financial news or article text here:",
                height=200,
                placeholder="e.g., 'The company reported strong quarterly earnings, exceeding analyst expectations...'"
            )
        else:
            user_text = st.text_area(
                "Example Text:",
                height=200,
                value="""The company announced impressive Q3 results with revenue growth exceeding analyst expectations. 
                Strong demand for new products and successful market expansion drove profitability. 
                Management raised full-year guidance, signaling confidence in sustained performance."""
            )
    
    if user_text.strip():
        with st.spinner("Analyzing sentiment..."):
            # Predict
            result = classifier.predict(user_text, return_probabilities=show_probabilities)
        
        # Results Section
        st.divider()
        st.subheader("📍 Analysis Results")
        
        col1, col2, col3 = st.columns(3)
        
        # Sentiment label
        sentiment = result["label"]
        confidence = result["confidence"]
        
        color_map = {
            "positive": "🟢",
            "negative": "🔴",
            "neutral": "🟡"
        }
        
        with col1:
            st.metric(
                "Sentiment",
                f"{color_map[sentiment]} {sentiment.upper()}",
                help="Predicted sentiment classification"
            )
        
        with col2:
            st.metric(
                "Confidence",
                f"{confidence*100:.1f}%",
                help="Model's confidence in the prediction"
            )
        
        with col3:
            st.metric(
                "Input Length",
                f"{len(user_text.split())} words",
                help="Number of words in input text"
            )
        
        # Probability Distribution
        if show_probabilities and "probabilities" in result:
            st.subheader("Probability Distribution")
            
            probs = result["probabilities"]
            labels = ["Positive", "Negative", "Neutral"]
            values = [probs["positive"], probs["negative"], probs["neutral"]]
            colors = ["#09AB3B", "#FF2B2B", "#FF9500"]
            
            # Horizontal bar chart
            fig = go.Figure(data=[
                go.Bar(
                    x=values,
                    y=labels,
                    orientation='h',
                    marker=dict(color=colors),
                    text=[f"{v*100:.1f}%" for v in values],
                    textposition='auto',
                )
            ])
            
            fig.update_layout(
                xaxis_title="Probability",
                yaxis_title="Sentiment Class",
                height=300,
                showlegend=False,
                xaxis=dict(range=[0, 1])
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Raw output (optional)
        if show_raw_output:
            st.subheader("Raw Output")
            st.json(result)
    else:
        st.info("👆 Enter text above to analyze sentiment")

# Tab 2: Batch Analysis
with tab2:
    st.subheader("Batch Text Analysis")
    
    batch_input = st.text_area(
        "Enter multiple texts (one per line):",
        height=200,
        placeholder="""Line 1: Strong earnings beat market expectations!
Line 2: Company faces significant challenges ahead.
Line 3: Quarterly results were in line with projections."""
    )
    
    if batch_input.strip():
        texts = [t.strip() for t in batch_input.split('\n') if t.strip()]
        
        if st.button("Analyze Batch", type="primary"):
            with st.spinner(f"Analyzing {len(texts)} texts..."):
                results = classifier.predict_batch(texts, return_probabilities=True)
            
            # Display results as table
            st.subheader("Results")
            
            table_data = []
            for i, result in enumerate(results, 1):
                table_data.append({
                    "✓": i,
                    "Text": result["text"][:50] + "..." if len(result["text"]) > 50 else result["text"],
                    "Sentiment": result["label"].upper(),
                    "Confidence": f"{result['confidence']*100:.1f}%"
                })
            
            st.dataframe(table_data, use_container_width=True)
            
            # Summary statistics
            st.subheader("Summary Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            sentiments = [r["label"] for r in results]
            
            with col1:
                positive_count = sentiments.count("positive")
                st.metric("Positive", positive_count)
            
            with col2:
                negative_count = sentiments.count("negative")
                st.metric("Negative", negative_count)
            
            with col3:
                neutral_count = sentiments.count("neutral")
                st.metric("Neutral", neutral_count)
            
            with col4:
                avg_confidence = sum(r["confidence"] for r in results) / len(results)
                st.metric("Avg Confidence", f"{avg_confidence*100:.1f}%")
            
            # Pie chart
            st.subheader("Sentiment Distribution")
            fig = go.Figure(data=[go.Pie(
                labels=["Positive", "Negative", "Neutral"],
                values=[positive_count, negative_count, neutral_count],
                marker=dict(colors=["#09AB3B", "#FF2B2B", "#FF9500"])
            )])
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("👆 Enter texts above to analyze")

# Footer
st.divider()
st.caption("💡 Financial Sentiment Analysis Tool | Powered by BERT + Financial Lexicon (XLex)")
