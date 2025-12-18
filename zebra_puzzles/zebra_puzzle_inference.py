#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zebra Puzzle Inference Script

Performs inference on zebra puzzles using Gemma-3-4B-IT and evaluates correctness.

Usage:
    python zebra_puzzle_inference.py <json_path> <num_puzzles>
    python zebra_puzzle_inference.py <json_path> <num_puzzles> --split test

Arguments:
    json_path       Path to zebra puzzles JSON file
    num_puzzles     Number of puzzles to process

Options:
    --split         Which split to use: 'train' or 'test' (default: train)

Example:
    python zebra_puzzle_inference.py zebra_puzzles_2x3_5rh_20251218_135450.json 10
    python zebra_puzzle_inference.py zebra_puzzles_2x3_5rh_20251218_135450.json 50 --split test
"""

import json
import os
import re
import argparse
from typing import Dict, List, Any, Optional
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class ZebraPuzzleModel:
    """Wrapper for Gemma-3-4B-IT for zebra puzzle inference."""
    
    def __init__(self, model_name: str = "google/gemma-3-4b-it", device: str = "auto"):
        self.model_name = model_name
        self.device = self._setup_device(device)
        self.tokenizer = None
        self.model = None
    
    def _setup_device(self, device: str) -> str:
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return device
    
    def load_model(self):
        """Load the model and tokenizer."""
        print(f"Loading {self.model_name} on {self.device}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        if self.device == "cuda":
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                trust_remote_code=True
            ).to(self.device)
        
        print(f"Model loaded successfully on {self.device}")
    
    def generate_response(self, prompt: str, max_new_tokens: int = 1000) -> str:
        """Generate a response for the given prompt."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.1,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.0
            )
        
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return response


def format_puzzle_prompt(puzzle: Dict[str, Any]) -> str:
    """Format a zebra puzzle as a prompt for the model."""
    prompt = puzzle['introduction']
    
    # Add clues
    for clue in puzzle['clues']:
        prompt += clue + "\n"
    
    prompt += "\n"
    prompt += puzzle['question'] + "\n\n"
    prompt += puzzle['format_instructions'] + "\n\n"
    prompt += "Format example:\n"
    prompt += puzzle['format_example'] + "\n\n"
    prompt += "Your answer:"
    
    return prompt


def extract_json_from_response(response: str) -> Optional[Dict]:
    """Extract JSON dictionary from model response."""
    # Try to find JSON in the response
    # Look for content between { and }
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(json_pattern, response, re.DOTALL)
    
    for match in matches:
        try:
            parsed = json.loads(match)
            # Check if it looks like a valid solution (has object_1, object_2, etc.)
            if any(key.startswith('object_') for key in parsed.keys()):
                return parsed
        except json.JSONDecodeError:
            continue
    
    # Try a more aggressive extraction
    try:
        # Find the first { and last }
        start = response.find('{')
        end = response.rfind('}')
        if start != -1 and end != -1 and end > start:
            candidate = response[start:end+1]
            return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    
    return None


def check_solution(predicted: Optional[Dict], expected: Dict) -> Dict[str, Any]:
    """
    Check if the predicted solution matches the expected solution.
    
    Returns dict with:
    - is_correct: bool - full match
    - partial_score: float - fraction of correct assignments
    - details: dict - per-house correctness
    """
    if predicted is None:
        return {
            'is_correct': False,
            'partial_score': 0.0,
            'details': {'error': 'Could not parse prediction'},
            'predicted': None,
            'expected': expected
        }
    
    total_correct = 0
    total_items = 0
    details = {}
    
    for house_key, expected_attrs in expected.items():
        if house_key not in predicted:
            details[house_key] = {'match': False, 'reason': 'missing'}
            total_items += len(expected_attrs)
            continue
        
        predicted_attrs = predicted[house_key]
        
        # Normalize for comparison (lowercase, strip)
        expected_normalized = [str(a).lower().strip() for a in expected_attrs]
        predicted_normalized = [str(a).lower().strip() for a in predicted_attrs] if isinstance(predicted_attrs, list) else []
        
        house_correct = 0
        for i, exp_attr in enumerate(expected_normalized):
            if i < len(predicted_normalized) and predicted_normalized[i] == exp_attr:
                house_correct += 1
        
        total_correct += house_correct
        total_items += len(expected_attrs)
        
        details[house_key] = {
            'match': predicted_normalized == expected_normalized,
            'expected': expected_attrs,
            'predicted': predicted_attrs if isinstance(predicted_attrs, list) else predicted.get(house_key),
            'correct_count': house_correct,
            'total': len(expected_attrs)
        }
    
    partial_score = total_correct / total_items if total_items > 0 else 0.0
    is_correct = partial_score == 1.0
    
    return {
        'is_correct': is_correct,
        'partial_score': partial_score,
        'details': details,
        'predicted': predicted,
        'expected': expected
    }


def load_puzzles(json_path: str, split: str = 'train') -> List[Dict]:
    """Load puzzles from JSON file."""
    print(f"\nLoading puzzles from: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    puzzles = data['data'].get(split, [])
    print(f"Loaded {len(puzzles)} puzzles from '{split}' split")
    
    return puzzles


def run_inference(json_path: str, num_puzzles: int, split: str = 'train',
                  output_dir: str = ".") -> str:
    """Run inference on zebra puzzles."""
    
    print("=" * 80)
    print("ZEBRA PUZZLE INFERENCE - Gemma-3-4B-IT")
    print("=" * 80)
    print(f"Number of puzzles: {num_puzzles}")
    print(f"Split: {split}")
    print("=" * 80)
    
    # Load puzzles
    puzzles = load_puzzles(json_path, split)
    
    if num_puzzles > len(puzzles):
        print(f"Warning: Requested {num_puzzles} puzzles but only {len(puzzles)} available")
        num_puzzles = len(puzzles)
    
    # Select puzzles
    selected_puzzles = puzzles[:num_puzzles]
    
    # Load model
    model = ZebraPuzzleModel(device="auto")
    model.load_model()
    
    # Process puzzles
    results = []
    correct_count = 0
    total_partial_score = 0.0
    
    for idx, puzzle in enumerate(selected_puzzles):
        print(f"\n{'='*80}")
        print(f"PUZZLE {idx + 1}/{num_puzzles}")
        print("=" * 80)
        
        # Format prompt
        prompt = format_puzzle_prompt(puzzle)
        
        # Show puzzle info
        num_clues = len(puzzle['clues'])
        num_red_herrings = len(puzzle.get('red_herrings', []))
        print(f"Clues: {num_clues} (including {num_red_herrings} red herrings)")
        
        # Generate response
        print("Generating response...")
        try:
            response = model.generate_response(prompt)
            print(f"Response preview: {response[:200]}...")
            
            # Extract and check solution
            predicted = extract_json_from_response(response)
            check_result = check_solution(predicted, puzzle['solution'])
            
            if check_result['is_correct']:
                correct_count += 1
                print("Result: CORRECT")
            else:
                print(f"Result: INCORRECT (partial score: {check_result['partial_score']:.2%})")
            
            total_partial_score += check_result['partial_score']
            
            # Store result
            result = {
                'puzzle_idx': idx,
                'num_clues': num_clues,
                'num_red_herrings': num_red_herrings,
                'red_herring_indices': puzzle.get('red_herrings', []),
                'clue_types': puzzle.get('clue_types', []),
                'prompt': prompt,
                'raw_response': response,
                'predicted_solution': check_result['predicted'],
                'expected_solution': puzzle['solution'],
                'is_correct': check_result['is_correct'],
                'partial_score': check_result['partial_score'],
                'check_details': check_result['details']
            }
            
            results.append(result)
            
        except Exception as e:
            print(f"Error processing puzzle: {e}")
            results.append({
                'puzzle_idx': idx,
                'error': str(e)
            })
    
    # Generate output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"zebra_inference_{split}_n{num_puzzles}_{timestamp}.json"
    output_path = os.path.join(output_dir, filename)
    
    # Calculate summary statistics
    accuracy = correct_count / num_puzzles if num_puzzles > 0 else 0
    avg_partial = total_partial_score / num_puzzles if num_puzzles > 0 else 0
    
    # Prepare output data
    output_data = {
        'metadata': {
            'model_name': 'google/gemma-3-4b-it',
            'inference_date': datetime.now().isoformat(),
            'source_file': json_path,
            'split': split,
            'num_puzzles': num_puzzles
        },
        'summary': {
            'total_puzzles': num_puzzles,
            'correct': correct_count,
            'accuracy': accuracy,
            'average_partial_score': avg_partial
        },
        'results': results
    }
    
    # Save results
    print(f"\nSaving results to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n" + "=" * 80)
    print("INFERENCE COMPLETE")
    print("=" * 80)
    print(f"Total puzzles: {num_puzzles}")
    print(f"Correct: {correct_count}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Average partial score: {avg_partial:.2%}")
    print(f"Results saved to: {output_path}")
    print("=" * 80)
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Run inference on zebra puzzles using Gemma-3-4B-IT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process 10 puzzles from training set
    python zebra_puzzle_inference.py zebra_puzzles_2x3_5rh_20251218_135450.json 10

    # Process 50 puzzles from test set
    python zebra_puzzle_inference.py zebra_puzzles_2x3_5rh_20251218_135450.json 50 --split test
        """
    )
    
    parser.add_argument('json_path', type=str,
                        help='Path to zebra puzzles JSON file')
    parser.add_argument('num_puzzles', type=int,
                        help='Number of puzzles to process')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'test'],
                        help="Which split to use: 'train' or 'test' (default: train)")
    parser.add_argument('--output-dir', type=str, default='.',
                        help='Output directory for results (default: current directory)')
    
    args = parser.parse_args()
    
    try:
        run_inference(
            json_path=args.json_path,
            num_puzzles=args.num_puzzles,
            split=args.split,
            output_dir=args.output_dir
        )
    except KeyboardInterrupt:
        print("\n\nInference interrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

