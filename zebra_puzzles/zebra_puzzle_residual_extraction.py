#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zebra Puzzle Inference with Residual Stream Extraction

Performs inference on zebra puzzles using Gemma-3-4B-IT, evaluates correctness,
and captures residual stream hidden states at layer 18.

Usage:
    python zebra_puzzle_residual_extraction.py <json_path> <num_puzzles> <num_tokens_after_prompt>
    python zebra_puzzle_residual_extraction.py <json_path> <num_puzzles> <num_tokens_after_prompt> --split test

Arguments:
    json_path               Path to zebra puzzles JSON file
    num_puzzles             Number of puzzles to process
    num_tokens_after_prompt Number of tokens to capture after prompt ends

Options:
    --split                 Which split to use: 'train' or 'test' (default: train)

Example:
    python zebra_puzzle_residual_extraction.py zebra_puzzles_2x3_5rh_20251218_135450.json 10 1
    python zebra_puzzle_residual_extraction.py zebra_puzzles_2x3_5rh_20251218_135450.json 50 5 --split test
"""

import json
import os
import re
import argparse
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM


class ZebraPuzzleResidualExtractor:
    """Wrapper for Gemma-3-4B-IT for zebra puzzle inference with residual stream extraction."""
    
    def __init__(self, model_name: str = "google/gemma-3-4b-it", device: str = "auto"):
        self.model_name = model_name
        self.device = self._setup_device(device)
        self.tokenizer = None
        self.model = None
        self.prompt_token_count = 0
        self._num_layers = None
        self._hidden_size = None
    
    def _setup_device(self, device: str) -> str:
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return device
    
    def _get_config_attr(self, config, attr_names: list, default=None):
        """Get a config attribute, trying multiple possible names."""
        for name in attr_names:
            if hasattr(config, name):
                return getattr(config, name)
        
        if hasattr(config, 'text_config'):
            for name in attr_names:
                if hasattr(config.text_config, name):
                    return getattr(config.text_config, name)
        
        return default
    
    def load_model(self):
        """Load the model and tokenizer with hidden states output enabled."""
        print(f"Loading {self.model_name} on {self.device}...")
        print("This may take a few minutes...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        if self.device == "cuda":
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                output_hidden_states=True
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                trust_remote_code=True,
                output_hidden_states=True
            ).to(self.device)
        
        self._num_layers = self._get_config_attr(
            self.model.config,
            ['num_hidden_layers', 'num_layers', 'n_layer', 'n_layers']
        )
        self._hidden_size = self._get_config_attr(
            self.model.config,
            ['hidden_size', 'hidden_dim', 'd_model', 'model_dim']
        )
        
        print(f"Model loaded successfully on {self.device}")
        print(f"Model has {self._num_layers} transformer layers")
        print(f"Hidden size: {self._hidden_size}")
    
    def generate_response_with_hidden_states(self, prompt: str, num_tokens_after_prompt: int,
                                              max_new_tokens: int = 1000) -> Dict[str, Any]:
        """
        Generate a response and capture hidden states for specified tokens.
        
        Args:
            prompt: Input prompt
            num_tokens_after_prompt: Number of tokens to capture after prompt ends
            max_new_tokens: Maximum tokens to generate
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        self.prompt_token_count = inputs['input_ids'].shape[1]
        
        # Calculate token range - start at last prompt token
        start_pos = self.prompt_token_count - 1
        end_pos = self.prompt_token_count - 1 + num_tokens_after_prompt - 1
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.1,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.0,
                output_hidden_states=True,
                return_dict_in_generate=True
            )
        
        # Extract generated token IDs
        generated_ids = outputs.sequences[0]
        
        # Decode response only (new tokens)
        response = self.tokenizer.decode(
            generated_ids[self.prompt_token_count:],
            skip_special_tokens=True
        )
        
        # Validate range
        actual_end_pos = end_pos
        if end_pos >= len(generated_ids):
            actual_end_pos = len(generated_ids) - 1
        
        # Extract hidden states for layer 18
        hidden_states = self._extract_hidden_states_for_range(
            outputs, start_pos, actual_end_pos, len(generated_ids)
        )
        
        return {
            'response': response,
            'prompt_token_count': self.prompt_token_count,
            'total_token_count': len(generated_ids),
            'token_range': (start_pos, actual_end_pos),
            'hidden_states': hidden_states
        }
    
    def _extract_hidden_states_for_range(self, outputs, start_position: int, end_position: int,
                                         total_tokens: int) -> Dict[str, Any]:
        """
        Extract hidden states from the generation output for a specific token range.
        
        Only captures layer 18 (midpoint of the 34-layer Gemma-3-4B model).
        """
        if not hasattr(outputs, 'hidden_states') or outputs.hidden_states is None:
            raise ValueError("No hidden states available in output")
        
        # Only capture layer 18
        layers_to_capture = [18]
        
        # Initialize storage for hidden states
        layer_hidden_states = {f"layer_{i}": {} for i in layers_to_capture}
        
        prompt_len = self.prompt_token_count
        first_token_in_step_0 = False
        
        for step_idx, step_hidden_states in enumerate(outputs.hidden_states):
            current_step_token_count = step_hidden_states[0].shape[1]
            
            if step_idx == 0:
                token_positions = list(range(current_step_token_count))
                first_token_in_step_0 = (current_step_token_count > prompt_len)
            else:
                if first_token_in_step_0:
                    token_positions = [prompt_len + step_idx]
                else:
                    token_positions = [prompt_len + step_idx - 1]
            
            # Only iterate over the layers we want to capture
            for layer_idx in layers_to_capture:
                layer_hidden = step_hidden_states[layer_idx]
                
                for local_idx, global_pos in enumerate(token_positions):
                    if start_position <= global_pos <= end_position:
                        if local_idx < layer_hidden.shape[1]:
                            hidden_vector = layer_hidden[0, local_idx, :].cpu().float().numpy()
                            layer_key = f"layer_{layer_idx}"
                            layer_hidden_states[layer_key][global_pos] = hidden_vector.tolist()
        
        return layer_hidden_states


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
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(json_pattern, response, re.DOTALL)
    
    for match in matches:
        try:
            parsed = json.loads(match)
            if any(key.startswith('object_') for key in parsed.keys()):
                return parsed
        except json.JSONDecodeError:
            continue
    
    # Try a more aggressive extraction
    try:
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


def run_inference_with_residuals(json_path: str, num_puzzles: int, 
                                  num_tokens_after_prompt: int,
                                  split: str = 'train',
                                  output_dir: str = ".") -> str:
    """Run inference on zebra puzzles with residual stream extraction."""
    
    print("=" * 80)
    print("ZEBRA PUZZLE INFERENCE WITH RESIDUAL STREAM EXTRACTION")
    print("=" * 80)
    print(f"Number of puzzles: {num_puzzles}")
    print(f"Tokens after prompt: {num_tokens_after_prompt}")
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
    extractor = ZebraPuzzleResidualExtractor(device="auto")
    extractor.load_model()
    
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
        
        # Generate response with hidden states
        print("Generating response and extracting hidden states...")
        try:
            gen_result = extractor.generate_response_with_hidden_states(
                prompt, num_tokens_after_prompt
            )
            
            response = gen_result['response']
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
                'check_details': check_result['details'],
                'prompt_token_count': gen_result['prompt_token_count'],
                'total_token_count': gen_result['total_token_count'],
                'token_range': gen_result['token_range'],
                'residual_stream_hidden_states': gen_result['hidden_states']
            }
            
            results.append(result)
            
        except Exception as e:
            print(f"Error processing puzzle: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'puzzle_idx': idx,
                'error': str(e)
            })
    
    # Generate output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"zebra_residual_{split}_n{num_puzzles}_tokens{num_tokens_after_prompt}_{timestamp}.json"
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
            'num_puzzles': num_puzzles,
            'num_tokens_after_prompt': num_tokens_after_prompt,
            'layers_captured': [18],
            'hidden_size': extractor._hidden_size
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
    
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"File size: {file_size:.2f} MB")
    
    # Print summary
    print("\n" + "=" * 80)
    print("INFERENCE COMPLETE")
    print("=" * 80)
    print(f"Total puzzles: {num_puzzles}")
    print(f"Correct: {correct_count}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Average partial score: {avg_partial:.2%}")
    print(f"Tokens captured per puzzle: {num_tokens_after_prompt}")
    print(f"Layer captured: 18")
    print(f"Results saved to: {output_path}")
    print("=" * 80)
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Run inference on zebra puzzles with residual stream extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process 10 puzzles, capture 1 token after prompt
    python zebra_puzzle_residual_extraction.py zebra_puzzles_2x3_5rh_20251218_135450.json 10 1

    # Process 50 puzzles from test set, capture 5 tokens
    python zebra_puzzle_residual_extraction.py zebra_puzzles_2x3_5rh_20251218_135450.json 50 5 --split test
        """
    )
    
    parser.add_argument('json_path', type=str,
                        help='Path to zebra puzzles JSON file')
    parser.add_argument('num_puzzles', type=int,
                        help='Number of puzzles to process')
    parser.add_argument('num_tokens_after_prompt', type=int,
                        help='Number of tokens to capture after prompt ends')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'test'],
                        help="Which split to use: 'train' or 'test' (default: train)")
    parser.add_argument('--output-dir', type=str, default='.',
                        help='Output directory for results (default: current directory)')
    
    args = parser.parse_args()
    
    try:
        run_inference_with_residuals(
            json_path=args.json_path,
            num_puzzles=args.num_puzzles,
            num_tokens_after_prompt=args.num_tokens_after_prompt,
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

