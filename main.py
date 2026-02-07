"""
Entry point script for the llmbothub application.
This allows running the app directly from the project root.
"""
import os
os.environ["OTEL_SDK_DISABLED"] = "true"

from llmbothub.main import main

if __name__ == "__main__":
    main()
