"""
Data Cleaning Script for Bulk Residual Stream Extraction Results

This script performs regex matching on raw_response text to extract
"Final Answer: {Letter}" patterns and saves the results to a JSON file.
"""

import json
import re
import argparse
from datetime import datetime
from pathlib import Path


def extract_final_answers(raw_response: str) -> dict:
    """
    Search for "Final Answer: {Letter}" patterns in the raw response.
    
    Matches patterns like:
    - "Final Answer: A\n"
    - "Final Answer: A*"
    - "Final Answer: A." 
    - "Final Answer: A " (with trailing space or end of string)
    
    Args:
        raw_response: The model's raw response text
        
    Returns:
        dict with regex_match, num_regex_matches, answer_letter, all_matches
    """
    # Pattern matches "Final Answer:" followed by optional whitespace, 
    # then a letter A-D, followed by newline, asterisk, period, space, or end of string
    pattern = r'Final Answer:\s*([ABCD])(?:\n|\*|\.|\s|$)'
    
    matches = re.findall(pattern, raw_response, re.IGNORECASE)
    
    # Normalize matches to uppercase
    matches = [m.upper() for m in matches]
    
    num_matches = len(matches)
    
    if num_matches == 0:
        return {
            "regex_match": False,
            "num_regex_matches": 0,
            "answer_letter": "N.A.",
            "all_matches": []
        }
    elif num_matches == 1:
        return {
            "regex_match": True,
            "num_regex_matches": 1,
            "answer_letter": matches[0],
            "all_matches": matches
        }
    else:
        # Multiple matches - check if they're all the same
        unique_matches = list(set(matches))
        if len(unique_matches) == 1:
            # All matches are the same letter
            return {
                "regex_match": True,
                "num_regex_matches": num_matches,
                "answer_letter": unique_matches[0],
                "all_matches": matches
            }
        else:
            # Different letters found - take the last one but flag for review
            return {
                "regex_match": True,
                "num_regex_matches": num_matches,
                "answer_letter": matches[-1],  # Use last match
                "all_matches": matches
            }


def process_bulk_extraction(input_path: str, output_path: str = None) -> dict:
    """
    Process all questions in the bulk extraction JSON file.
    
    Args:
        input_path: Path to the bulk extraction JSON file
        output_path: Optional path for output JSON (auto-generated if not provided)
        
    Returns:
        dict containing all results and summary statistics
    """
    print(f"Loading data from: {input_path}")
    
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    results = []
    
    # Statistics
    total_questions = 0
    regex_matched = 0
    multiple_matches = 0
    no_match = 0
    answer_distribution = {"A": 0, "B": 0, "C": 0, "D": 0, "N.A.": 0}
    
    for item in data['results']:
        question_idx = item['question_idx']
        raw_response = item.get('raw_response', '')
        
        # Extract final answer using regex
        extraction = extract_final_answers(raw_response)
        
        result = {
            "question_idx": question_idx,
            "regex_match": extraction["regex_match"],
            "num_regex_matches": extraction["num_regex_matches"],
            "answer_letter": extraction["answer_letter"]
        }
        
        # Add all_matches only if there are multiple different matches (for review)
        if extraction["num_regex_matches"] > 1 and len(set(extraction["all_matches"])) > 1:
            result["all_matches"] = extraction["all_matches"]
            result["needs_manual_review"] = True
        
        results.append(result)
        
        # Update statistics
        total_questions += 1
        if extraction["regex_match"]:
            regex_matched += 1
        else:
            no_match += 1
        
        if extraction["num_regex_matches"] > 1:
            multiple_matches += 1
        
        answer_distribution[extraction["answer_letter"]] += 1
    
    # Create output
    output_data = {
        "metadata": {
            "source_file": str(input_path),
            "processing_date": datetime.now().isoformat(),
            "total_questions": total_questions,
            "regex_pattern": r'Final Answer:\s*([ABCD])(?:\n|\*|\.|\s|$)'
        },
        "statistics": {
            "total_questions": total_questions,
            "regex_matched": regex_matched,
            "no_match": no_match,
            "multiple_matches_count": multiple_matches,
            "answer_distribution": answer_distribution
        },
        "results": results
    }
    
    # Generate output path if not provided
    if output_path is None:
        input_file = Path(input_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = input_file.parent / f"cleaned_answers_{timestamp}.json"
    
    # Save results
    print(f"\nSaving results to: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)
    print(f"Total questions processed: {total_questions}")
    print(f"Regex matches found: {regex_matched} ({100*regex_matched/total_questions:.1f}%)")
    print(f"No match found: {no_match} ({100*no_match/total_questions:.1f}%)")
    print(f"Questions with multiple matches: {multiple_matches}")
    print("\nAnswer Distribution:")
    for letter, count in answer_distribution.items():
        pct = 100*count/total_questions if total_questions > 0 else 0
        print(f"  {letter}: {count} ({pct:.1f}%)")
    print("="*60)
    
    # List questions needing manual review
    review_needed = [r for r in results if r.get("needs_manual_review", False)]
    if review_needed:
        print(f"\n⚠️  {len(review_needed)} questions need manual review (multiple different matches):")
        for r in review_needed[:10]:  # Show first 10
            print(f"  Question {r['question_idx']}: matches = {r['all_matches']}")
        if len(review_needed) > 10:
            print(f"  ... and {len(review_needed) - 10} more")
    
    return output_data


def main():
    parser = argparse.ArgumentParser(
        description="Extract 'Final Answer: {Letter}' patterns from bulk extraction results"
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the bulk extraction JSON file"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output JSON file path (auto-generated if not provided)"
    )
    
    args = parser.parse_args()
    
    process_bulk_extraction(args.input_file, args.output)


if __name__ == "__main__":
    main()

