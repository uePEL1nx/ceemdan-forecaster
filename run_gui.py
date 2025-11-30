#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Launch the Gradio GUI for CEEMDAN-Informer-LSTM Pipeline.

Usage:
    python run_gui.py
    # Opens browser to http://localhost:7860

Options:
    --port PORT     Port to run on (default: 7860)
    --share         Create a public Gradio link
    --no-browser    Don't open browser automatically
"""
import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from gui.app import create_app


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Launch the CEEMDAN Pipeline GUI"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run on (default: 7860)"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio link"
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't open browser automatically"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("CEEMDAN-Informer-LSTM Pipeline GUI")
    print("=" * 60)
    print(f"Starting on http://localhost:{args.port}")
    if args.share:
        print("Creating public share link...")
    print("=" * 60)

    app = create_app()
    app.launch(
        server_name="127.0.0.1",
        server_port=args.port,
        share=args.share,
        inbrowser=not args.no_browser,
        show_error=True,
    )


if __name__ == "__main__":
    main()
