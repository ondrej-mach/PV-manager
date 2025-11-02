"""Backward-compat shim.

Use train.py for training and predict.py for inference.
This file remains to avoid breaking existing entrypoints.
"""

import importlib

if __name__ == "__main__":
    train = importlib.import_module("train")
    train.main()