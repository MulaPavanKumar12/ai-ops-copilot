"""Allow ``python -m aiops_copilot.generator generate``."""

import sys

from aiops_copilot.generator.cli import main

if __name__ == "__main__":
    sys.exit(main())
