# src/orcanvas/cli.py
import os
import sys
from typing import NoReturn

def main():
    print("Open Research Canvas")


def _exit(code: int) -> NoReturn:
    sys.exit(code)
    
if __name__ == "__main__":
    _exit(main())