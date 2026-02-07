"""
Entry point script for the llm-bawt application.
This allows running the app directly from the project root.
"""
import os
os.environ["OTEL_SDK_DISABLED"] = "true"

from llm_bawt.main import main

if __name__ == "__main__":
    main()
