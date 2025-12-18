#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download Zebra Puzzles Dataset

Downloads the zebra puzzles dataset from Hugging Face and saves it locally as JSON.

Dataset: alexandrainst/zebra_puzzles
Config: dataset_en_houses_2x3_5rh

Usage:
    python download_zebra_puzzles.py
"""

import json
import os
from datetime import datetime
from datasets import load_dataset


def download_zebra_puzzles(output_dir: str = "."):
    """
    Download the zebra puzzles dataset and save as JSON.
    
    Args:
        output_dir: Directory to save the output JSON file
    """
    print("=" * 80)
    print("DOWNLOADING ZEBRA PUZZLES DATASET")
    print("=" * 80)
    
    dataset_name = "alexandrainst/zebra_puzzles"
    config_name = "dataset_en_houses_2x3_5rh"
    
    print(f"\nDataset: {dataset_name}")
    print(f"Config: {config_name}")
    
    # Load dataset
    print("\nLoading dataset from Hugging Face...")
    dataset = load_dataset(dataset_name, config_name)
    
    print(f"\nDataset structure:")
    print(f"  Splits: {list(dataset.keys())}")
    
    # Prepare output data
    output_data = {
        "metadata": {
            "dataset_name": dataset_name,
            "config": config_name,
            "download_date": datetime.now().isoformat(),
            "splits": {}
        },
        "data": {}
    }
    
    # Process each split
    total_samples = 0
    for split_name in dataset.keys():
        split_data = dataset[split_name]
        num_samples = len(split_data)
        total_samples += num_samples
        
        print(f"\n  {split_name}: {num_samples} samples")
        
        # Get column names
        if num_samples > 0:
            print(f"    Columns: {split_data.column_names}")
        
        output_data["metadata"]["splits"][split_name] = num_samples
        
        # Convert to list of dicts
        output_data["data"][split_name] = [dict(sample) for sample in split_data]
    
    print(f"\nTotal samples across all splits: {total_samples}")
    
    # Generate output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"zebra_puzzles_2x3_5rh_{timestamp}.json"
    output_path = os.path.join(output_dir, filename)
    
    # Save to JSON
    print(f"\nSaving to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"File size: {file_size:.2f} MB")
    
    # Print sample
    if total_samples > 0:
        first_split = list(dataset.keys())[0]
        first_sample = output_data["data"][first_split][0]
        print(f"\nSample entry from '{first_split}':")
        print("-" * 40)
        for key, value in first_sample.items():
            if isinstance(value, str) and len(value) > 200:
                print(f"  {key}: {value[:200]}...")
            else:
                print(f"  {key}: {value}")
    
    print("\n" + "=" * 80)
    print("DOWNLOAD COMPLETE!")
    print("=" * 80)
    
    return output_path


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    download_zebra_puzzles(output_dir=script_dir)


if __name__ == "__main__":
    main()

