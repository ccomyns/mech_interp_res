#!/usr/bin/env python3
"""
data_cleaning_manual.py

Script to extract manual answers from a manual review text file and update
the corresponding cleaned_answers JSON file, replacing "N.A." entries with
the manually provided answers.

Usage:
    python data_cleaning_manual.py <manual_review_txt> <cleaned_answers_json>

Example:
    python data_cleaning_manual.py \
        data_collection_linear_probe/manual_review_20251207_195706.txt \
        data_collection_linear_probe/cleaned_answers_first_half.json
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple


def parse_manual_review_file(txt_path: str) -> Dict[int, str]:
    """
    Parse the manual review text file and extract question indices and their manual answers.
    
    Args:
        txt_path: Path to the manual review text file
        
    Returns:
        Dictionary mapping question_idx to manual answer letter
    """
    manual_answers = {}
    
    with open(txt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by the separator lines to get individual question blocks
    # Pattern to match question blocks with their index
    question_pattern = r'Question \d+ of \d+ \(Index: (\d+)\)'
    answer_pattern = r'MANUAL ANSWER:\s*([ABCD]|NONE|N\.?A\.?)?'
    
    # Find all question indices
    question_matches = list(re.finditer(question_pattern, content))
    
    for i, match in enumerate(question_matches):
        question_idx = int(match.group(1))
        
        # Get the text block for this question (until next question or end)
        start_pos = match.end()
        if i + 1 < len(question_matches):
            end_pos = question_matches[i + 1].start()
        else:
            end_pos = len(content)
        
        question_block = content[start_pos:end_pos]
        
        # Find the manual answer in this block
        answer_match = re.search(answer_pattern, question_block, re.IGNORECASE)
        
        if answer_match:
            answer = answer_match.group(1)
            if answer:
                answer = answer.strip().upper()
                # Only store valid answers (A, B, C, D)
                if answer in ['A', 'B', 'C', 'D']:
                    manual_answers[question_idx] = answer
                elif answer in ['NONE', 'NA', 'N.A.', 'N.A']:
                    # Explicitly marked as no answer - keep as N.A.
                    pass
                # If empty or whitespace, it hasn't been filled in yet
    
    return manual_answers


def update_cleaned_answers(json_path: str, manual_answers: Dict[int, str]) -> Tuple[int, int, int]:
    """
    Update the cleaned answers JSON file with manual answers.
    
    Args:
        json_path: Path to the cleaned answers JSON file
        manual_answers: Dictionary mapping question_idx to manual answer letter
        
    Returns:
        Tuple of (total_updated, total_na_remaining, total_results)
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    updated_count = 0
    na_remaining = 0
    
    # Track statistics changes
    old_stats = data.get('statistics', {}).copy()
    
    # Update results
    for result in data['results']:
        question_idx = result['question_idx']
        
        if result['answer_letter'] == 'N.A.' and question_idx in manual_answers:
            result['answer_letter'] = manual_answers[question_idx]
            result['manual_override'] = True  # Mark that this was manually updated
            updated_count += 1
        elif result['answer_letter'] == 'N.A.':
            na_remaining += 1
    
    # Recalculate statistics
    answer_distribution = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'N.A.': 0}
    correct_answer_count = 0
    agreed_with_hint_count = 0
    incorrect_answer_count = 0
    
    for result in data['results']:
        answer = result['answer_letter']
        if answer in answer_distribution:
            answer_distribution[answer] += 1
        
        if answer != 'N.A.':
            if answer == result['correct_answer_letter']:
                correct_answer_count += 1
            elif answer == result['incorrect_hint']:
                agreed_with_hint_count += 1
            else:
                incorrect_answer_count += 1
    
    # Update statistics in data
    data['statistics']['answer_distribution'] = answer_distribution
    data['statistics']['correct_answer'] = correct_answer_count
    data['statistics']['agreed_with_hint'] = agreed_with_hint_count
    data['statistics']['incorrect_answer'] = incorrect_answer_count
    
    # Add metadata about manual update
    if 'manual_update_info' not in data:
        data['manual_update_info'] = []
    
    data['manual_update_info'].append({
        'update_date': datetime.now().isoformat(),
        'entries_updated': updated_count,
        'na_remaining': na_remaining
    })
    
    # Write updated data back to file
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    
    return updated_count, na_remaining, len(data['results'])


def main():
    parser = argparse.ArgumentParser(
        description='Extract manual answers from review file and update cleaned answers JSON',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python data_cleaning_manual.py \\
        data_collection_linear_probe/manual_review_20251207_195706.txt \\
        data_collection_linear_probe/cleaned_answers_first_half.json
        """
    )
    
    parser.add_argument(
        'manual_review_txt',
        type=str,
        help='Path to the manual review text file containing filled-in answers'
    )
    
    parser.add_argument(
        'cleaned_answers_json',
        type=str,
        help='Path to the cleaned answers JSON file to update'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be updated without making changes'
    )
    
    args = parser.parse_args()
    
    # Validate input files
    if not Path(args.manual_review_txt).exists():
        print(f"Error: Manual review file not found: {args.manual_review_txt}")
        sys.exit(1)
    
    if not Path(args.cleaned_answers_json).exists():
        print(f"Error: Cleaned answers JSON not found: {args.cleaned_answers_json}")
        sys.exit(1)
    
    # Parse manual answers
    print(f"\n📖 Reading manual answers from: {args.manual_review_txt}")
    manual_answers = parse_manual_review_file(args.manual_review_txt)
    
    print(f"   Found {len(manual_answers)} manual answers")
    
    if args.dry_run:
        print("\n🔍 DRY RUN - No changes will be made")
        print(f"\nManual answers found:")
        for idx, answer in sorted(manual_answers.items()):
            print(f"   Question {idx}: {answer}")
        return
    
    # Update cleaned answers
    print(f"\n📝 Updating: {args.cleaned_answers_json}")
    updated, remaining, total = update_cleaned_answers(args.cleaned_answers_json, manual_answers)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"   Total questions:        {total}")
    print(f"   Manual answers found:   {len(manual_answers)}")
    print(f"   Entries updated:        {updated}")
    print(f"   N.A. remaining:         {remaining}")
    print("=" * 60)
    
    if updated > 0:
        print(f"\n✅ Successfully updated {updated} entries in {args.cleaned_answers_json}")
    else:
        print("\n⚠️  No entries were updated. Make sure manual answers are filled in the review file.")


if __name__ == '__main__':
    main()

