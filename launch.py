#!/usr/bin/env python3
"""Quick-start launcher for Financial Sentiment Analysis frontends."""

import subprocess
import sys
import os
from pathlib import Path

def check_model():
    """Check if model exists."""
    model_path = Path("models_checkpoint/best_model.pt")
    if not model_path.exists():
        print("⚠️  Model checkpoint not found!")
        print(f"   Path: {model_path}")
        print("\nTo train the model, run: python main.py")
        print("or check SETUP_AND_RUN.md for instructions.\n")
        return False
    return True


def check_dependencies():
    """Check if all dependencies are installed."""
    packages = ['streamlit', 'flask', 'torch', 'transformers', 'plotly']
    missing = []

    for package in packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)

    if missing:
        print("⚠️  Missing packages:")
        for pkg in missing:
            print(f"   - {pkg}")
        print("\nTo install, run: pip install -r requirements.txt\n")
        return False
    return True


def main():
    """Main menu."""
    print("\n" + "="*60)
    print("🚀 Financial Sentiment Analysis - Launcher")
    print("="*60)

    # Check model
    if not check_model():
        print("❌ Cannot proceed without model.")
        sys.exit(1)

    # Check dependencies
    if not check_dependencies():
        print("❌ Please install missing dependencies.")
        sys.exit(1)

    print("\n✓ All checks passed!\n")
    print("Choose a frontend:")
    print("  1️⃣  Streamlit Web App (Recommended for users)")
    print("  2️⃣  Flask REST API (For developers/integration)")
    print("  3️⃣  HTML Web Frontend (Lightweight, uses API)")
    print("  4️⃣  CLI Tool (Command-line interface)")
    print("  5️⃣  Run All (Start API, then open HTML in browser)")
    print("  6️⃣  Exit")

    choice = input("\nSelect (1-6): ").strip()

    if choice == '1':
        launch_streamlit()
    elif choice == '2':
        launch_api()
    elif choice == '3':
        launch_html()
    elif choice == '4':
        launch_cli()
    elif choice == '5':
        launch_all()
    elif choice == '6':
        print("Goodbye! 👋\n")
        sys.exit(0)
    else:
        print("Invalid choice!")
        sys.exit(1)


def launch_streamlit():
    """Launch Streamlit app."""
    print("\n" + "="*60)
    print("🎯 Launching Streamlit App...")
    print("="*60)
    print("\n✓ Opening http://localhost:8501")
    print("⚠️  Press Ctrl+C to stop the server\n")

    try:
        subprocess.run(['streamlit', 'run', 'app.py'], check=True)
    except KeyboardInterrupt:
        print("\n\n✓ Stopped.")
    except FileNotFoundError:
        print("❌ Streamlit not found. Install with: pip install streamlit")


def launch_api():
    """Launch Flask API."""
    print("\n" + "="*60)
    print("🔌 Launching Flask API Server...")
    print("="*60)
    print("\n✓ API running on http://localhost:5000")
    print("📚 API Documentation:")
    print("   GET  /health")
    print("   GET  /api/v1/models")
    print("   POST /api/v1/predict")
    print("   POST /api/v1/predict-batch")
    print("\n💡 Open http://localhost:5000/health in browser to verify")
    print("⚠️  Press Ctrl+C to stop the server\n")

    try:
        subprocess.run(['python', 'api.py'], check=True)
    except KeyboardInterrupt:
        print("\n\n✓ Stopped.")


def launch_html():
    """Launch HTML frontend with API."""
    print("\n" + "="*60)
    print("🌐 Launching HTML Web Frontend...")
    print("="*60)

    # Check if API is running
    print("\n⚠️  This frontend requires the Flask API to be running.")
    print("\nDo you want to:")
    print("  1. Start the API first, then open the HTML file")
    print("  2. Open the HTML file only (API must already be running)")
    print("  3. Cancel")

    choice = input("\nSelect (1-3): ").strip()

    if choice == '1':
        print("\n✓ Starting Flask API...")
        api_process = subprocess.Popen(['python', 'api.py'])
        print("✓ API started")

        import time
        time.sleep(2)

        print("✓ Opening index.html in browser...")
        html_path = Path("index.html").resolve()
        webbrowser_url = f"file://{html_path}"

        try:
            import webbrowser
            webbrowser.open(webbrowser_url)
            print(f"✓ Opened {webbrowser_url}")
            print("\n⚠️  Press Ctrl+C to stop both servers\n")
            api_process.wait()
        except KeyboardInterrupt:
            print("\n✓ Stopping API...")
            api_process.terminate()
            api_process.wait()
            print("✓ Stopped.")

    elif choice == '2':
        print("✓ Opening index.html in browser...")
        html_path = Path("index.html").resolve()

        try:
            import webbrowser
            webbrowser.open(f"file://{html_path}")
            print(f"✓ Opened {html_path}")
            print("\n💡 Make sure you have the API running:")
            print("   python api.py")
        except Exception as e:
            print(f"❌ Error: {e}")
    else:
        print("Cancelled.")


def launch_cli():
    """Launch CLI tool in interactive mode."""
    print("\n" + "="*60)
    print("⌨️  CLI Interactive Mode")
    print("="*60)
    print("\nEnter financial texts to analyze.")
    print("Type 'quit' or 'exit' to exit\n")

    try:
        subprocess.run(['python', 'cli.py'], check=True)
    except KeyboardInterrupt:
        print("\n\n✓ Stopped.")
    except FileNotFoundError:
        print("❌ CLI not found.")


def launch_all():
    """Start API and open HTML in browser."""
    print("\n" + "="*60)
    print("🚀 Starting All Components...")
    print("="*60)

    print("\n1️⃣  Starting Flask API on http://localhost:5000...")
    api_process = subprocess.Popen(['python', 'api.py'],
                                   stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL)

    import time
    time.sleep(3)

    print("2️⃣  Opening HTML frontend in browser...")
    html_path = Path("index.html").resolve()

    try:
        import webbrowser
        webbrowser.open(f"file://{html_path}")
        print(f"✓ Frontend opened at file://{html_path}")
        print("\n✓ Both services are running!")
        print("   API: http://localhost:5000")
        print("   Frontend: file://" + str(html_path))
        print("\n⚠️  Press Ctrl+C to stop all services\n")

        api_process.wait()
    except KeyboardInterrupt:
        print("\n\n✓ Stopping services...")
        api_process.terminate()
        api_process.wait()
        print("✓ All services stopped.")
    except Exception as e:
        print(f"❌ Error: {e}")
        api_process.terminate()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
