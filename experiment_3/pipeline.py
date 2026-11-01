#!/usr/bin/env python3
from e3.cli import main

if __name__ == '__main__':
    import sys
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
    except (ValueError, KeyError, OSError, ImportError, RuntimeError, FloatingPointError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        raise SystemExit(2)
