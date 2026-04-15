"""Command-line interface for Financial Sentiment Analysis."""

import argparse
import json
import sys
from pathlib import Path
from inference import FinancialSentimentClassifier
from tabulate import tabulate
from colorama import Fore, Style, init

# Initialize colorama for cross-platform colors
init(autoreset=True)


def format_sentiment(label, confidence):
    """Format sentiment with color coding."""
    colors = {
        "positive": Fore.GREEN,
        "negative": Fore.RED,
        "neutral": Fore.YELLOW
    }
    emoji = {
        "positive": "📈",
        "negative": "📉",
        "neutral": "➡️"
    }
    color = colors.get(label, "")
    icon = emoji.get(label, "")
    return f"{color}{icon} {label.upper()}{Style.RESET_ALL} ({confidence*100:.1f}%)"


def analyze_single(classifier, text, verbose=False, json_output=False):
    """Analyze single text."""
    result = classifier.predict(text, return_probabilities=True)
    
    if json_output:
        print(json.dumps(result, indent=2))
    else:
        print("\n" + "="*60)
        print("FINANCIAL SENTIMENT ANALYSIS")
        print("="*60)
        
        print(f"\n📝 Text: {text[:100]}..." if len(text) > 100 else f"\n📝 Text: {text}")
        print(f"\n🎯 Sentiment: {format_sentiment(result['label'], result['confidence'])}")
        
        if verbose:
            probs = result["probabilities"]
            print("\n📊 Probability Distribution:")
            table_data = [
                ["Positive", f"{probs['positive']*100:.2f}%"],
                ["Negative", f"{probs['negative']*100:.2f}%"],
                ["Neutral", f"{probs['neutral']*100:.2f}%"]
            ]
            print(tabulate(table_data, headers=["Class", "Probability"], tablefmt="grid"))
        
        print("\n" + "="*60 + "\n")


def analyze_batch(classifier, file_path, json_output=False, summary=True):
    """Analyze multiple texts from file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"{Fore.RED}✗ Error: File '{file_path}' not found{Style.RESET_ALL}")
        sys.exit(1)
    
    print(f"\n{Fore.CYAN}Analyzing {len(texts)} texts...{Style.RESET_ALL}")
    results = classifier.predict_batch(texts, return_probabilities=True)
    
    if json_output:
        print(json.dumps(results, indent=2))
    else:
        print("\n" + "="*80)
        print("BATCH ANALYSIS RESULTS")
        print("="*80 + "\n")
        
        table_data = []
        for i, result in enumerate(results, 1):
            text_preview = result["text"][:40] + "..." if len(result["text"]) > 40 else result["text"]
            table_data.append([
                i,
                text_preview,
                result["label"].upper(),
                f"{result['confidence']*100:.1f}%"
            ])
        
        print(tabulate(
            table_data,
            headers=["#", "Text", "Sentiment", "Confidence"],
            tablefmt="grid"
        ))
        
        if summary:
            print("\n" + "-"*80)
            print("SUMMARY STATISTICS")
            print("-"*80)
            
            sentiments = [r["label"] for r in results]
            positive = sentiments.count("positive")
            negative = sentiments.count("negative")
            neutral = sentiments.count("neutral")
            avg_confidence = sum(r["confidence"] for r in results) / len(results)
            
            summary_data = [
                ["Total Texts", len(results)],
                ["Positive", positive],
                ["Negative", negative],
                ["Neutral", neutral],
                ["Average Confidence", f"{avg_confidence*100:.1f}%"]
            ]
            print(tabulate(summary_data, tablefmt="grid"))
        
        print("\n" + "="*80 + "\n")


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Financial Sentiment Analysis CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Analyze single text:
    python cli.py "Strong earnings beat market expectations!"
  
  Analyze with probabilities:
    python cli.py "Company faces challenges ahead" --verbose
  
  Batch analysis from file:
    python cli.py --file articles.txt
  
  Output as JSON:
    python cli.py "Some text" --json > results.json
        """
    )
    
    parser.add_argument(
        'text',
        nargs='?',
        default=None,
        help='Text to analyze'
    )
    parser.add_argument(
        '-f', '--file',
        type=str,
        help='Analyze texts from file (one per line)'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Show detailed probability distribution'
    )
    parser.add_argument(
        '-j', '--json',
        action='store_true',
        help='Output as JSON'
    )
    parser.add_argument(
        '--no-summary',
        action='store_true',
        help='Skip summary statistics in batch mode'
    )
    parser.add_argument(
        '-m', '--model',
        type=str,
        default='models_checkpoint/best_model.pt',
        help='Path to model checkpoint'
    )
    
    args = parser.parse_args()
    
    # Load classifier
    if args.json is False:  # Only show loading message if not JSON output
        print(f"{Fore.CYAN}🔄 Loading model...{Style.RESET_ALL}", end=" ", flush=True)
    
    classifier = FinancialSentimentClassifier(model_path=args.model)
    
    if args.json is False:
        print(f"{Fore.GREEN}✓{Style.RESET_ALL}\n")
    
    # Analyze
    if args.file:
        # Batch mode
        analyze_batch(classifier, args.file, json_output=args.json, summary=not args.no_summary)
    elif args.text:
        # Single text mode
        analyze_single(classifier, args.text, verbose=args.verbose, json_output=args.json)
    else:
        # Interactive mode
        print(f"{Fore.CYAN}Financial Sentiment Analysis CLI{Style.RESET_ALL}")
        print("Type 'quit' or 'exit' to exit\n")
        
        while True:
            try:
                text = input(f"{Fore.BLUE}Enter text:{Style.RESET_ALL} ").strip()
                if text.lower() in ['quit', 'exit']:
                    print(f"{Fore.YELLOW}Goodbye!{Style.RESET_ALL}")
                    break
                if text:
                    analyze_single(classifier, text, verbose=False)
            except KeyboardInterrupt:
                print(f"\n{Fore.YELLOW}Interrupted. Goodbye!{Style.RESET_ALL}")
                break
            except Exception as e:
                print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")


if __name__ == "__main__":
    main()
