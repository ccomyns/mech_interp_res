"""
Categorize Linear Probe Training Data

This script takes two cleaned answer JSON files and categorizes questions into:
1. correct_answer: Questions where the model selected the correct answer
2. agrees_with_hint: Questions where the model agreed with the incorrect hint

For each category, the first 200 questions are labeled as 'train' and 
the remaining are labeled as 'test'.
"""

import json
import argparse
from datetime import datetime
from pathlib import Path


def categorize_questions(input_file1: str, input_file2: str, output_file: str = None):
    """
    Categorize questions from two cleaned answer files.
    
    Args:
        input_file1: Path to first cleaned answers JSON
        input_file2: Path to second cleaned answers JSON
        output_file: Optional path for output JSON (auto-generated if not provided)
    """
    print(f"Loading data from: {input_file1}")
    with open(input_file1, 'r') as f:
        data1 = json.load(f)
    
    print(f"Loading data from: {input_file2}")
    with open(input_file2, 'r') as f:
        data2 = json.load(f)
    
    # Combine results from both files
    all_results = data1['results'] + data2['results']
    
    print(f"Total questions: {len(all_results)}")
    
    # Categorize questions
    correct_answer_list = []
    agrees_with_hint_list = []
    
    for result in all_results:
        question_idx = result['question_idx']
        answer_letter = result['answer_letter']
        correct_answer_letter = result['correct_answer_letter']
        incorrect_hint = result['incorrect_hint']
        
        # Skip N.A. answers
        if answer_letter == "N.A.":
            continue
        
        # Categorize based on answer
        if answer_letter == correct_answer_letter:
            correct_answer_list.append({
                "question_idx": question_idx,
                "answer_letter": answer_letter,
                "correct_answer_letter": correct_answer_letter,
                "incorrect_hint": incorrect_hint
            })
        elif answer_letter == incorrect_hint:
            agrees_with_hint_list.append({
                "question_idx": question_idx,
                "answer_letter": answer_letter,
                "correct_answer_letter": correct_answer_letter,
                "incorrect_hint": incorrect_hint
            })
    
    # Add train/test labels
    # First 200 in each category are 'train', rest are 'test'
    for i, item in enumerate(correct_answer_list):
        if i < 200:
            item['split'] = 'train'
        else:
            item['split'] = 'test'
    
    for i, item in enumerate(agrees_with_hint_list):
        if i < 200:
            item['split'] = 'train'
        else:
            item['split'] = 'test'
    
    # Count train/test splits
    correct_train = sum(1 for x in correct_answer_list if x['split'] == 'train')
    correct_test = sum(1 for x in correct_answer_list if x['split'] == 'test')
    hint_train = sum(1 for x in agrees_with_hint_list if x['split'] == 'train')
    hint_test = sum(1 for x in agrees_with_hint_list if x['split'] == 'test')
    
    # Create output structure
    output_data = {
        "metadata": {
            "source_files": [str(input_file1), str(input_file2)],
            "processing_date": datetime.now().isoformat(),
            "description": "Questions categorized by whether model selected correct answer or agreed with incorrect hint. First 200 of each category are train, rest are test."
        },
        "statistics": {
            "total_correct_answer": len(correct_answer_list),
            "total_agrees_with_hint": len(agrees_with_hint_list),
            "correct_answer_train": correct_train,
            "correct_answer_test": correct_test,
            "agrees_with_hint_train": hint_train,
            "agrees_with_hint_test": hint_test
        },
        "correct_answer": correct_answer_list,
        "agrees_with_hint": agrees_with_hint_list
    }
    
    # Generate output path if not provided
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = Path(input_file1).parent / f"categorized_data_{timestamp}.json"
    
    # Save output
    print(f"\nSaving results to: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("CATEGORIZATION SUMMARY")
    print("="*60)
    print(f"Total questions categorized: {len(correct_answer_list) + len(agrees_with_hint_list)}")
    print("\nCorrect Answer:")
    print(f"  Total: {len(correct_answer_list)}")
    print(f"  Train: {correct_train}")
    print(f"  Test: {correct_test}")
    print("\nAgrees with Hint:")
    print(f"  Total: {len(agrees_with_hint_list)}")
    print(f"  Train: {hint_train}")
    print(f"  Test: {hint_test}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Categorize questions into correct_answer and agrees_with_hint for linear probe training"
    )
    parser.add_argument(
        "first_half",
        type=str,
        help="Path to the first cleaned answers JSON file"
    )
    parser.add_argument(
        "second_half",
        type=str,
        help="Path to the second cleaned answers JSON file"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output JSON file path (auto-generated if not provided)"
    )
    
    args = parser.parse_args()
    
    categorize_questions(args.first_half, args.second_half, args.output)


if __name__ == "__main__":
    main()

