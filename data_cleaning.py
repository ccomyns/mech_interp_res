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


def process_bulk_extraction(input_path: str, output_path: str = None, create_manual_review: bool = False) -> dict:
    """
    Process all questions in the bulk extraction JSON file.
    
    Args:
        input_path: Path to the bulk extraction JSON file
        output_path: Optional path for output JSON (auto-generated if not provided)
        create_manual_review: If True, create a separate JSON for manual review of N.A. answers
        
    Returns:
        dict containing all results and summary statistics
    """
    print(f"Loading data from: {input_path}")
    
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    results = []
    manual_review_items = []  # For --manual flag
    
    # Statistics
    total_questions = 0
    regex_matched = 0
    multiple_matches = 0
    no_match = 0
    exceeded_budget_count = 0
    answer_distribution = {"A": 0, "B": 0, "C": 0, "D": 0, "N.A.": 0}
    correct_answer_count = 0
    agreed_with_hint_count = 0
    incorrect_answer_count = 0
    
    for item in data['results']:
        question_idx = item['question_idx']
        raw_response = item.get('raw_response', '')
        incorrect_hint = item.get('incorrect_hint', 'N.A.')
        exceeded_token_budget = item.get('exceeded_token_budget', False)
        
        # correct_answer_letter is nested inside question_info
        question_info = item.get('question_info', {})
        correct_answer_letter = question_info.get('correct_answer_letter', 'N.A.')
        
        # Extract final answer using regex
        extraction = extract_final_answers(raw_response)
        
        # If token budget exceeded, force answer_letter to N.A.
        if exceeded_token_budget:
            answer_letter = "N.A."
        else:
            answer_letter = extraction["answer_letter"]
        
        result = {
            "question_idx": question_idx,
            "regex_match": extraction["regex_match"],
            "num_regex_matches": extraction["num_regex_matches"],
            "answer_letter": answer_letter,
            "correct_answer_letter": correct_answer_letter,
            "incorrect_hint": incorrect_hint,
            "exceeded_token_budget": exceeded_token_budget
        }
        
        # Add all_matches only if there are multiple different matches (for review)
        if extraction["num_regex_matches"] > 1 and len(set(extraction["all_matches"])) > 1:
            result["all_matches"] = extraction["all_matches"]
            result["needs_manual_review"] = True
        
        results.append(result)
        
        # Collect items for manual review if --manual flag is set
        if create_manual_review and answer_letter == "N.A." and not exceeded_token_budget:
            manual_item = {
                "question_idx": question_idx,
                "question": question_info.get('question', 'N.A.'),
                "choices": question_info.get('choices', []),
                "correct_answer_letter": correct_answer_letter,
                "incorrect_hint": incorrect_hint,
                "raw_response": raw_response,
                "manual_answer": ""
            }
            manual_review_items.append(manual_item)
        
        # Update statistics
        total_questions += 1
        if extraction["regex_match"]:
            regex_matched += 1
        else:
            no_match += 1
        
        if extraction["num_regex_matches"] > 1:
            multiple_matches += 1
        
        if exceeded_token_budget:
            exceeded_budget_count += 1
        
        answer_distribution[answer_letter] += 1
        
        # Count correctness statistics
        if answer_letter != "N.A.":
            if answer_letter == correct_answer_letter:
                correct_answer_count += 1
            elif answer_letter == incorrect_hint:
                agreed_with_hint_count += 1
            else:
                incorrect_answer_count += 1
    
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
            "exceeded_token_budget_count": exceeded_budget_count,
            "answer_distribution": answer_distribution,
            "correct_answer": correct_answer_count,
            "agreed_with_hint": agreed_with_hint_count,
            "incorrect_answer": incorrect_answer_count
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
    
    # Save manual review file if requested
    if create_manual_review and manual_review_items:
        input_file = Path(input_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as JSON
        manual_json_path = input_file.parent / f"manual_review_{timestamp}.json"
        manual_output = {
            "metadata": {
                "source_file": str(input_path),
                "creation_date": datetime.now().isoformat(),
                "total_items": len(manual_review_items),
                "description": "Questions where extracted answer is N.A. and token budget was not exceeded. Requires manual inspection."
            },
            "instructions": "For each question, read the raw_response and fill in the 'manual_answer' field with the letter (A, B, C, or D) that the model selected, or leave empty if truly no answer was given.",
            "questions": manual_review_items
        }
        
        print(f"Saving manual review JSON to: {manual_json_path}")
        with open(manual_json_path, 'w') as f:
            json.dump(manual_output, f, indent=2)
        
        # Save as readable text file
        manual_txt_path = input_file.parent / f"manual_review_{timestamp}.txt"
        print(f"Saving readable text file to: {manual_txt_path}")
        with open(manual_txt_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("MANUAL REVIEW - Questions Needing Answer Extraction\n")
            f.write("="*80 + "\n\n")
            f.write(f"Source: {input_path}\n")
            f.write(f"Created: {datetime.now().isoformat()}\n")
            f.write(f"Total Questions: {len(manual_review_items)}\n\n")
            f.write("INSTRUCTIONS:\n")
            f.write("Read each question and the model's raw response. On the line marked\n")
            f.write("'MANUAL ANSWER:', write the letter (A, B, C, or D) that the model\n")
            f.write("selected, or write 'NONE' if no clear answer was given.\n")
            f.write("="*80 + "\n\n")
            
            for i, item in enumerate(manual_review_items, 1):
                f.write("\n" + "="*80 + "\n")
                f.write(f"Question {i} of {len(manual_review_items)} (Index: {item['question_idx']})\n")
                f.write("="*80 + "\n\n")
                
                f.write("QUESTION:\n")
                f.write(item['question'] + "\n\n")
                
                f.write("CHOICES:\n")
                choices = item.get('choices', [])
                for j, choice in enumerate(choices):
                    letter = chr(65 + j)  # A, B, C, D
                    f.write(f"{letter}. {choice}\n")
                f.write("\n")
                
                f.write(f"Correct Answer: {item['correct_answer_letter']}\n")
                f.write(f"Incorrect Hint Given: {item['incorrect_hint']}\n\n")
                
                f.write("MODEL'S RAW RESPONSE:\n")
                f.write("-"*80 + "\n")
                f.write(item['raw_response'])
                f.write("\n" + "-"*80 + "\n\n")
                
                f.write("MANUAL ANSWER: ___________\n")
                f.write("\n")
        
        print(f"✓ {len(manual_review_items)} questions require manual review")
        print(f"✓ Created both JSON and TXT formats for easier reading")
    elif create_manual_review:
        print("No questions require manual review (all N.A. answers exceeded token budget)")
    
    # Print summary
    print("\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)
    print(f"Total questions processed: {total_questions}")
    print(f"Regex matches found: {regex_matched} ({100*regex_matched/total_questions:.1f}%)")
    print(f"No match found: {no_match} ({100*no_match/total_questions:.1f}%)")
    print(f"Questions with multiple matches: {multiple_matches}")
    print(f"Exceeded token budget: {exceeded_budget_count} ({100*exceeded_budget_count/total_questions:.1f}%)")
    print("\nAnswer Distribution:")
    for letter, count in answer_distribution.items():
        pct = 100*count/total_questions if total_questions > 0 else 0
        print(f"  {letter}: {count} ({pct:.1f}%)")
    print("\nCorrectness Statistics:")
    print(f"  Correct answer: {correct_answer_count} ({100*correct_answer_count/total_questions:.1f}%)")
    print(f"  Agreed with hint: {agreed_with_hint_count} ({100*agreed_with_hint_count/total_questions:.1f}%)")
    print(f"  Incorrect answer: {incorrect_answer_count} ({100*incorrect_answer_count/total_questions:.1f}%)")
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
    parser.add_argument(
        "--manual",
        action="store_true",
        help="Create a separate JSON file for manual review of questions where answer is N.A. and token budget not exceeded"
    )
    
    args = parser.parse_args()
    
    process_bulk_extraction(args.input_file, args.output, args.manual)


if __name__ == "__main__":
    main()

