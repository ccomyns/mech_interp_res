"""
Categorize Linear Probe Training Data

This script takes two cleaned answer JSON files and categorizes questions into:
1. correct_answer: Questions where the model selected the correct answer
2. agrees_with_hint: Questions where the model agreed with the incorrect hint

For each category, the first 200 questions are labeled as 'train' and 
the remaining are labeled as 'test'.

Supports excluding specific subjects (e.g., math-related subjects).
"""

import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Set


# Default math-related subjects to exclude
MATH_SUBJECTS = [
    "abstract_algebra",
    "college_mathematics",
    "elementary_mathematics",
    "high_school_mathematics",
    "high_school_statistics",
    "econometrics"
]


def load_subject_mapping(mmlu_path: str) -> dict:
    """
    Load MMLU dataset and create a mapping from question index to subject.
    
    Args:
        mmlu_path: Path to mmlu_sample_dataset.json
        
    Returns:
        Dictionary mapping question_idx to subject name
    """
    with open(mmlu_path, 'r') as f:
        mmlu_data = json.load(f)
    
    subject_map = {}
    for idx, question in enumerate(mmlu_data['questions']):
        subject_map[idx] = question['subject']
    
    return subject_map


def categorize_questions(
    input_file1: str, 
    input_file2: str, 
    output_file: str = None,
    mmlu_path: str = None,
    exclude_subjects: List[str] = None
):
    """
    Categorize questions from two cleaned answer files.
    
    Args:
        input_file1: Path to first cleaned answers JSON
        input_file2: Path to second cleaned answers JSON
        output_file: Optional path for output JSON (auto-generated if not provided)
        mmlu_path: Path to MMLU dataset for subject lookup (required if excluding subjects)
        exclude_subjects: List of subjects to exclude from categorization
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
    
    # Load subject mapping if excluding subjects
    subject_map = None
    excluded_set: Set[str] = set()
    excluded_count = 0
    
    if exclude_subjects:
        if not mmlu_path:
            raise ValueError("--mmlu-dataset is required when using --exclude-subjects")
        
        print(f"\nLoading MMLU dataset from: {mmlu_path}")
        subject_map = load_subject_mapping(mmlu_path)
        excluded_set = set(exclude_subjects)
        print(f"Excluding subjects: {', '.join(sorted(excluded_set))}")
    
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
        
        # Skip excluded subjects
        if subject_map and question_idx in subject_map:
            subject = subject_map[question_idx]
            if subject in excluded_set:
                excluded_count += 1
                continue
        
        # Categorize based on answer
        if answer_letter == correct_answer_letter:
            entry = {
                "question_idx": question_idx,
                "answer_letter": answer_letter,
                "correct_answer_letter": correct_answer_letter,
                "incorrect_hint": incorrect_hint
            }
            # Include subject if available
            if subject_map and question_idx in subject_map:
                entry["subject"] = subject_map[question_idx]
            correct_answer_list.append(entry)
            
        elif answer_letter == incorrect_hint:
            entry = {
                "question_idx": question_idx,
                "answer_letter": answer_letter,
                "correct_answer_letter": correct_answer_letter,
                "incorrect_hint": incorrect_hint
            }
            # Include subject if available
            if subject_map and question_idx in subject_map:
                entry["subject"] = subject_map[question_idx]
            agrees_with_hint_list.append(entry)
    
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
    
    # Build description
    description = "Questions categorized by whether model selected correct answer or agreed with incorrect hint. First 200 of each category are train, rest are test."
    if exclude_subjects:
        description += f" Excluded subjects: {', '.join(sorted(excluded_set))}."
    
    # Create output structure
    output_data = {
        "metadata": {
            "source_files": [str(input_file1), str(input_file2)],
            "processing_date": datetime.now().isoformat(),
            "description": description,
            "excluded_subjects": sorted(list(excluded_set)) if exclude_subjects else [],
            "questions_excluded_by_subject": excluded_count
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
        suffix = "_no_math" if exclude_subjects else ""
        output_file = Path(input_file1).parent / f"categorized_data{suffix}_{timestamp}.json"
    
    # Save output
    print(f"\nSaving results to: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("CATEGORIZATION SUMMARY")
    print("="*60)
    
    if exclude_subjects:
        print(f"Questions excluded by subject: {excluded_count}")
        print(f"Excluded subjects: {', '.join(sorted(excluded_set))}")
        print()
    
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
        description="Categorize questions into correct_answer and agrees_with_hint for linear probe training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage (no subject filtering):
    python categorize_linear_probe_data.py \\
        data_collection_linear_probe/cleaned_answers_first_half.json \\
        data_collection_linear_probe/cleaned_answers_second_half.json

    # Exclude math-related subjects:
    python categorize_linear_probe_data.py \\
        data_collection_linear_probe/cleaned_answers_first_half.json \\
        data_collection_linear_probe/cleaned_answers_second_half.json \\
        --mmlu-dataset mmlu_sample_dataset.json \\
        --exclude-subjects abstract_algebra college_mathematics elementary_mathematics \\
            high_school_mathematics high_school_statistics econometrics

    # Use --exclude-math shortcut:
    python categorize_linear_probe_data.py \\
        data_collection_linear_probe/cleaned_answers_first_half.json \\
        data_collection_linear_probe/cleaned_answers_second_half.json \\
        --mmlu-dataset mmlu_sample_dataset.json \\
        --exclude-math
        """
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
    parser.add_argument(
        "--mmlu-dataset",
        type=str,
        default=None,
        help="Path to mmlu_sample_dataset.json (required when excluding subjects)"
    )
    parser.add_argument(
        "--exclude-subjects",
        type=str,
        nargs="+",
        default=None,
        help="List of subject names to exclude from categorization"
    )
    parser.add_argument(
        "--exclude-math",
        action="store_true",
        help=f"Shortcut to exclude common math subjects: {', '.join(MATH_SUBJECTS)}"
    )
    
    args = parser.parse_args()
    
    # Handle --exclude-math shortcut
    exclude_subjects = args.exclude_subjects
    if args.exclude_math:
        if exclude_subjects:
            # Combine with any explicitly provided subjects
            exclude_subjects = list(set(exclude_subjects + MATH_SUBJECTS))
        else:
            exclude_subjects = MATH_SUBJECTS.copy()
    
    # Validate: if excluding subjects, mmlu-dataset is required
    if exclude_subjects and not args.mmlu_dataset:
        parser.error("--mmlu-dataset is required when using --exclude-subjects or --exclude-math")
    
    categorize_questions(
        args.first_half, 
        args.second_half, 
        args.output,
        args.mmlu_dataset,
        exclude_subjects
    )


if __name__ == "__main__":
    main()
