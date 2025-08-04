#!/usr/bin/env python3
"""
Simple validation runner script
"""
import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from validation import main
from config_ssl_upload import ex

if __name__ == "__main__":
    print("Starting validation...")
    print("Loading trained model and running validation on test set...")
    
    # Run validation with the experiment configuration
    try:
        ex.run()
        print("Validation completed successfully!")
    except Exception as e:
        print(f"Validation failed with error: {e}")
        import traceback
        traceback.print_exc()