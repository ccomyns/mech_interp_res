#!/usr/bin/env python3
"""
Download misleading cognitive questions from the inverse-scaling dataset.

Dataset: inverse-scaling-ttc/inverse-scaling-ttc-main
Target: 2500 misleading cognitive questions

Usage:
    python download_misleading_cognitive.py
"""

import json
from datetime import datetime
from datasets import load_dataset

def download_misleading_cognitive():
    """Download and save misleading cognitive questions from inverse-scaling dataset."""
    
    print("=" * 80)
    print("DOWNLOADING MISLEADING COGNITIVE QUESTIONS")
    print("=" * 80)
    print("Dataset: inverse-scaling-ttc/inverse-scaling-ttc-main")
    print()
    
    # Load the dataset with the specific config for misleading cognitive biases
    print("📂 Loading dataset from Hugging Face...")
    print("   Config: misleading_cognitive_biases")
    dataset = load_dataset("inverse-scaling-ttc/inverse-scaling-ttc-main", "misleading_cognitive_biases")
    
    # Print available splits and info
    print(f"✅ Dataset loaded!")
    print(f"   Available splits: {list(dataset.keys())}")
    
    # Check the structure
    for split_name, split_data in dataset.items():
        print(f"   {split_name}: {len(split_data)} examples")
        if len(split_data) > 0:
            print(f"      Columns: {split_data.column_names}")
    
    # Get the main split (likely 'train' or 'test')
    main_split = list(dataset.keys())[0]
    filtered_data = dataset[main_split]
    matched_name = "misleading_cognitive_biases"
    
    print(f"\n✅ Found {len(filtered_data)} questions (config: {matched_name})")
    
    # Convert to list of dicts for saving
    print("\n📝 Converting to JSON format...")
    questions = []
    for i, item in enumerate(filtered_data):
        question_dict = {k: v for k, v in item.items()}
        question_dict['index'] = i
        questions.append(question_dict)
    
    # Print sample structure
    if questions:
        print(f"\n📋 Sample question structure:")
        sample = questions[0]
        for key, value in sample.items():
            if isinstance(value, str) and len(value) > 100:
                print(f"   {key}: {value[:100]}...")
            else:
                print(f"   {key}: {value}")
    
    # Save to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"misleading_cognitive_{len(questions)}_{timestamp}.json"
    
    output_data = {
        'metadata': {
            'source': 'inverse-scaling-ttc/inverse-scaling-ttc-main',
            'task_name': matched_name,
            'download_date': datetime.now().isoformat(),
            'num_questions': len(questions)
        },
        'questions': questions
    }
    
    print(f"\n💾 Saving to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved {len(questions)} questions")
    
    # Summary
    print("\n" + "=" * 80)
    print("DOWNLOAD COMPLETE")
    print("=" * 80)
    print(f"Output file: {output_file}")
    print(f"Total questions: {len(questions)}")
    print("=" * 80)
    
    return output_file


if __name__ == "__main__":
    try:
        download_misleading_cognitive()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

