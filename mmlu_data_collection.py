#!/usr/bin/env python3
"""
MMLU Data Collection Script

Downloads MMLU (Massive Multitask Language Understanding) questions from HuggingFace
and saves them to a local JSON file.

Usage:
    python3 mmlu_data_collection.py --output mmlu_dataset.json
    python3 mmlu_data_collection.py --output mmlu_dataset.json --subjects "high_school_physics,college_chemistry"
    python3 mmlu_data_collection.py --output mmlu_dataset.json --max-per-subject 50
    python3 mmlu_data_collection.py --output mmlu_dataset.json --split test
    python3 mmlu_data_collection.py --list-subjects
"""

import argparse
import json
from datetime import datetime
from typing import List, Optional
from datasets import load_dataset, get_dataset_config_names


def get_all_subjects() -> List[str]:
    """Get all available MMLU subjects from HuggingFace."""
    try:
        configs = get_dataset_config_names("cais/mmlu")
        # Filter out 'all' config if present
        subjects = [c for c in configs if c != 'all']
        return sorted(subjects)
    except Exception as e:
        print(f"Error fetching subject list: {e}")
        return []


def load_mmlu_questions(
    subjects: Optional[List[str]] = None,
    split: str = "test",
    max_per_subject: Optional[int] = None
) -> List[dict]:
    """
    Load MMLU questions from HuggingFace.
    
    Args:
        subjects: List of subject names to load. If None, loads all subjects.
        split: Dataset split to use ('test', 'validation', 'dev', 'auxiliary_train')
        max_per_subject: Maximum number of questions per subject. None for all.
    
    Returns:
        List of question dictionaries
    """
    all_questions = []
    
    # Get subjects to load
    if subjects is None:
        subjects = get_all_subjects()
    
    print(f"Loading MMLU questions from {len(subjects)} subjects...")
    print(f"Split: {split}")
    if max_per_subject:
        print(f"Max per subject: {max_per_subject}")
    print()
    
    for i, subject in enumerate(subjects):
        try:
            print(f"  [{i+1}/{len(subjects)}] Loading {subject}...", end=" ")
            
            # Load the specific subject
            dataset = load_dataset("cais/mmlu", subject, split=split, trust_remote_code=True)
            
            # Convert to list of dicts
            questions = []
            for idx, item in enumerate(dataset):
                if max_per_subject and idx >= max_per_subject:
                    break
                
                question_data = {
                    "subject": subject,
                    "question": item["question"],
                    "choices": item["choices"],
                    "answer_idx": item["answer"],  # 0-3 index
                    "answer_letter": chr(65 + item["answer"])  # A, B, C, or D
                }
                questions.append(question_data)
            
            all_questions.extend(questions)
            print(f"{len(questions)} questions")
            
        except Exception as e:
            print(f"Error loading {subject}: {e}")
            continue
    
    return all_questions


def save_questions(questions: List[dict], output_path: str, split: str):
    """Save questions to a JSON file with metadata."""
    
    # Count questions per subject
    subject_counts = {}
    for q in questions:
        subject = q["subject"]
        subject_counts[subject] = subject_counts.get(subject, 0) + 1
    
    output_data = {
        "metadata": {
            "source": "cais/mmlu",
            "split": split,
            "total_questions": len(questions),
            "num_subjects": len(subject_counts),
            "collected_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "subject_counts": subject_counts,
        "questions": questions
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved {len(questions)} questions to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Download MMLU questions from HuggingFace and save to JSON"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output JSON file path (default: mmlu_{split}_{timestamp}.json)"
    )
    parser.add_argument(
        "--subjects",
        type=str,
        default=None,
        help="Comma-separated list of subjects to download (default: all)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "validation", "dev", "auxiliary_train"],
        help="Dataset split to use (default: test)"
    )
    parser.add_argument(
        "--max-per-subject",
        type=int,
        default=None,
        help="Maximum number of questions per subject (default: all)"
    )
    parser.add_argument(
        "--list-subjects",
        action="store_true",
        help="List all available MMLU subjects and exit"
    )
    
    args = parser.parse_args()
    
    # Handle list-subjects flag
    if args.list_subjects:
        print("Available MMLU subjects:\n")
        subjects = get_all_subjects()
        for i, subject in enumerate(subjects, 1):
            print(f"  {i:3d}. {subject}")
        print(f"\nTotal: {len(subjects)} subjects")
        return
    
    # Parse subjects if provided
    subjects = None
    if args.subjects:
        subjects = [s.strip() for s in args.subjects.split(",")]
        print(f"Selected subjects: {subjects}")
    
    # Generate default output filename if not provided
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"mmlu_{args.split}_{timestamp}.json"
    
    # Load questions
    questions = load_mmlu_questions(
        subjects=subjects,
        split=args.split,
        max_per_subject=args.max_per_subject
    )
    
    if not questions:
        print("No questions loaded!")
        return
    
    # Save to file
    save_questions(questions, args.output, args.split)
    
    # Print summary
    print("\n" + "=" * 50)
    print("Summary:")
    print(f"  Total questions: {len(questions)}")
    print(f"  Split: {args.split}")
    print(f"  Output file: {args.output}")


if __name__ == "__main__":
    main()

