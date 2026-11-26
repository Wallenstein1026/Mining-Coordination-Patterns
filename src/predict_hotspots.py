"""Compatibility entry point for the canonical chronological forecast implementation."""
import sys
from run_experiments import main

if __name__ == '__main__':
    sys.argv.insert(1, 'forecast')
    main()
