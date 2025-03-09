#cache_utils.py
import os
import pandas as pd

# Simple in-memory cache
_dataset_cache = {}

def get_cached_dataset(data_path):
    """Get dataset from cache if available"""
    if data_path in _dataset_cache:
        return _dataset_cache[data_path]
    return None

def set_cached_dataset(data_path, df):
    """Store dataset in cache"""
    _dataset_cache[data_path] = df

def clear_cache() -> None:
    """Clear the dataset cache"""
    _dataset_cache.clear()

def get_file_extension(file_path):
    """Get file extension from file path"""
    if '.' not in file_path:
        return ''
    return file_path.split('.')[-1].lower() 