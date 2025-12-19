"""Allow running tracer as module: python -m pytest_fkit.tracer"""
import sys
from .cli import main

if __name__ == "__main__":
    sys.exit(main())
