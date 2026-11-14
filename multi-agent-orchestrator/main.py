"""Entry point: python main.py <command> ...  (same as the `orchestrator` console script)."""
import sys

from orchestrator.cli import main

if __name__ == "__main__":
    sys.exit(main())
