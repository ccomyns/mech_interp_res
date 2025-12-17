"""
Baseline Truth Marker

This script takes a baseline inference JSON file and a cleaned answers JSON file,
then adds a 'baseline_correct' field to each question in the cleaned answers file
based on whether the model answered correctly in the unbiased baseline.

Usage:
    python baseline_truth.py baseline_results.json cleaned_answers.json
"""

import json
import argparse
from datetime import datetime
from pathlib import Path


def load_baseline_data(baseline_path: str) -> tuple:
    """
    Load baseline results and return sets of question indices.
    
    Args:
        baseline_path: Path to baseline inference results JSON
        
    Returns:
        Tuple of (correct_indices, all_indices) where:
        - correct_indices: Set of question indices where is_correct is True
        - all_indices: Set of all question indices in the baseline file
    """
    print(f"Loading baseline results from: {baseline_path}")
    with open(baseline_path, 'r') as f:
        baseline_data = json.load(f)
    
    correct_indices = set()
    all_indices = set()
    
    for result in baseline_data['results']:
        question_idx = result['question_idx']
        all_indices.add(question_idx)
        is_correct = result.get('is_correct', False)
        
        if is_correct:
            correct_indices.add(question_idx)
    
    total_questions = len(all_indices)
    print(f"Total questions in baseline: {total_questions}")
    print(f"Questions answered correctly: {len(correct_indices)}")
    print(f"Accuracy: {100 * len(correct_indices) / total_questions:.2f}%")
    
    return correct_indices, all_indices


def add_baseline_field(cleaned_answers_path: str, correct_indices: set, all_indices: set, output_path: str = None):
    """
    Add 'baseline_correct' field to each question in cleaned answers JSON.
    Only updates questions that exist in the baseline file.
    
    Args:
        cleaned_answers_path: Path to cleaned answers JSON
        correct_indices: Set of question indices where baseline was correct
        all_indices: Set of all question indices in the baseline file
        output_path: Optional output path (defaults to overwriting input)
    """
    print(f"\nLoading cleaned answers from: {cleaned_answers_path}")
    with open(cleaned_answers_path, 'r') as f:
        cleaned_data = json.load(f)
    
    # Track statistics
    marked_correct = 0
    marked_incorrect = 0
    skipped = 0
    
    # Add baseline_correct field to each result (only if in baseline)
    for result in cleaned_data['results']:
        question_idx = result['question_idx']
        
        # Only update questions that exist in the baseline file
        if question_idx not in all_indices:
            skipped += 1
            continue
        
        if question_idx in correct_indices:
            result['baseline_correct'] = True
            marked_correct += 1
        else:
            result['baseline_correct'] = False
            marked_incorrect += 1
    
    # Add metadata about baseline update
    if 'baseline_update_info' not in cleaned_data:
        cleaned_data['baseline_update_info'] = []
    
    cleaned_data['baseline_update_info'].append({
        "update_date": datetime.now().isoformat(),
        "baseline_correct_count": marked_correct,
        "baseline_incorrect_count": marked_incorrect,
        "skipped_not_in_baseline": skipped
    })
    
    # Determine output path
    if output_path is None:
        output_path = cleaned_answers_path
    
    # Save updated data
    print(f"\nSaving updated cleaned answers to: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(cleaned_data, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("BASELINE TRUTH UPDATE SUMMARY")
    print("="*60)
    print(f"Questions marked baseline_correct=True:  {marked_correct}")
    print(f"Questions marked baseline_correct=False: {marked_incorrect}")
    print(f"Questions skipped (not in baseline):     {skipped}")
    print(f"Total questions updated: {marked_correct + marked_incorrect}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Add baseline_correct field to cleaned answers JSON based on baseline inference results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Update cleaned answers with baseline truth (overwrites original):
    python baseline_truth.py baseline_q0-569.json cleaned_answers_first_half.json

    # Save to a new file:
    python baseline_truth.py baseline_q0-569.json cleaned_answers_first_half.json -o cleaned_answers_with_baseline.json
        """
    )
    parser.add_argument(
        "baseline_json",
        type=str,
        help="Path to baseline inference results JSON file"
    )
    parser.add_argument(
        "cleaned_answers_json",
        type=str,
        help="Path to cleaned answers JSON file to update"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output path for updated JSON (default: overwrite input file)"
    )
    
    args = parser.parse_args()
    
    # Load baseline data (correct indices and all indices)
    correct_indices, all_indices = load_baseline_data(args.baseline_json)
    
    # Add baseline_correct field to cleaned answers (only for questions in baseline)
    add_baseline_field(args.cleaned_answers_json, correct_indices, all_indices, args.output)


if __name__ == "__main__":
    main()

