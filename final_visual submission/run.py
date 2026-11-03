"""Terminal entry point. Importing this file never starts an experiment."""
from src.cli import main

if __name__ == '__main__':
    raise SystemExit(main())
