"""Utility to load a pickled object from a specified file path."""

import pickle


def load_object(filepath: str):
    """Load a pickled object from a file."""
    with open(filepath, "rb") as f:
        return pickle.load(f)
