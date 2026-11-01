#!/usr/bin/env python3
"""Command-line entrypoint; keep this entire project separate from encoder projects."""
import subprocess
import sys
from e2b.cli import main

if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:raise SystemExit(130)
    except (OSError,ValueError,KeyError,RuntimeError,ImportError,FloatingPointError,subprocess.CalledProcessError) as exc:
        print(f'ERROR: {exc}',file=sys.stderr)
        raise SystemExit(2)
