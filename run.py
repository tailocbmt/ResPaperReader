#!/usr/bin/env python3
import os
import sys
import subprocess
import argparse
import dotenv


def main():
    """Main entry point for the Research Paper Reader application."""
    # Load environment variables from .env file if it exists
    dotenv.load_dotenv()

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Research Paper Reader and Assistant")
    parser.add_argument("--port", type=int, default=8501,
                        help="Port to run the Streamlit app on")
    parser.add_argument("--gemini-key", type=str, help="Google Gemini API key")
    args = parser.parse_args()

    # Set API key if provided
    if args.gemini_key:
        os.environ["GEMINI_API_KEY"] = args.gemini_key

    # Add environment variable to fix Torch errors in Streamlit's watcher
    os.environ["PYTHONPATH"] = os.path.dirname(os.path.abspath(__file__))

    # Ensure data directories exist
    os.makedirs("data/uploads", exist_ok=True)

    # Run the Streamlit app
    cmd = [
        "streamlit", "run",
        "app/app.py",
        "--server.port", str(args.port),
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false"
    ]

    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nShutting down Research Paper Reader...")
        sys.exit(0)


if __name__ == "__main__":
    main()
