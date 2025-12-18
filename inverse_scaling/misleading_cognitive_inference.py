#!/usr/bin/env python3
"""
Inference script for misleading cognitive questions using Gemma-3-4B-IT.

Runs inference on questions from the inverse-scaling misleading cognitive dataset,
evaluates whether the model gives the correct answer, and captures residual stream
hidden states at layer 18.

Usage:
    python misleading_cognitive_inference.py <json_path> <start_idx> <end_idx> <num_tokens_after_prompt>

Arguments:
    json_path              Path to misleading cognitive questions JSON file
    start_idx              Starting question index (inclusive)
    end_idx                Ending question index (exclusive)
    num_tokens_after_prompt Number of tokens to capture after prompt ends

Example:
    python misleading_cognitive_inference.py misleading_cognitive_2500_20251218_113450.json 0 10 1
    # Processes questions 0-9, captures 1 token after prompt (the first generated token)
    
    python misleading_cognitive_inference.py misleading_cognitive_2500_20251218_113450.json 0 100 5
    # Processes questions 0-99, captures 5 tokens after prompt
"""

import json
import os
import re
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM


class GemmaInference:
    """Inference class for Gemma-3-4B-IT model with residual stream extraction."""
    
    def __init__(self, model_name: str = "google/gemma-3-4b-it", device: str = "auto"):
        self.model_name = model_name
        self.device = self._setup_device(device)
        self.tokenizer = None
        self.model = None
        self.prompt_token_count = 0
    
    def _setup_device(self, device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def load_model(self):
        """Load the Gemma model and tokenizer."""
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
        
        print(f"✅ Model loaded successfully on {self.device}")
    
    def generate_response_with_hidden_states(self, prompt: str, num_tokens_after_prompt: int = 1,
                                              max_new_tokens: int = 500) -> Dict[str, Any]:
        """Generate a response and capture hidden states for specified tokens.
        
        Args:
            prompt: Input prompt
            num_tokens_after_prompt: Number of tokens to capture after the prompt ends
            max_new_tokens: Maximum tokens to generate
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        self.prompt_token_count = inputs['input_ids'].shape[1]
        
        # Calculate token range - start at last prompt token, capture num_tokens_after_prompt
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
        
        # Decode only the new tokens
        response = self.tokenizer.decode(
            generated_ids[self.prompt_token_count:],
            skip_special_tokens=True
        )
        
        # Extract hidden states for layer 18
        hidden_states = self._extract_hidden_states_for_range(
            outputs, start_pos, end_pos
        )
        
        return {
            'response': response,
            'prompt_token_count': self.prompt_token_count,
            'total_token_count': len(generated_ids),
            'token_range': (start_pos, end_pos),
            'hidden_states': hidden_states
        }
    
    def _extract_hidden_states_for_range(self, outputs, start_position: int, end_position: int) -> Dict[str, Any]:
        """Extract hidden states from layer 18 for a specific token range."""
        if not hasattr(outputs, 'hidden_states') or outputs.hidden_states is None:
            raise ValueError("No hidden states available in output")
        
        # Only capture layer 18 (midpoint of Gemma-3-4B's 34 layers)
        layers_to_capture = [18]
        
        # Initialize storage
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
            
            for layer_idx in layers_to_capture:
                layer_hidden = step_hidden_states[layer_idx]
                
                for local_idx, global_pos in enumerate(token_positions):
                    if start_position <= global_pos <= end_position:
                        if local_idx < layer_hidden.shape[1]:
                            hidden_vector = layer_hidden[0, local_idx, :].cpu().float().numpy()
                            layer_key = f"layer_{layer_idx}"
                            layer_hidden_states[layer_key][str(global_pos)] = hidden_vector.tolist()
        
        return layer_hidden_states


def extract_answer(response: str) -> Optional[int]:
    """Extract the integer answer from <answer> tags in the response."""
    # Try to find <answer>NUMBER</answer> pattern
    match = re.search(r'<answer>\s*(\d+)\s*</answer>', response, re.IGNORECASE)
    if match:
        return int(match.group(1))
    
    # Try alternate patterns
    match = re.search(r'<answer>\s*(-?\d+)\s*</answer>', response, re.IGNORECASE)
    if match:
        return int(match.group(1))
    
    # Try finding just a number after "answer" keyword
    match = re.search(r'answer[:\s]+(\d+)', response, re.IGNORECASE)
    if match:
        return int(match.group(1))
    
    return None


def load_questions(json_path: str) -> List[Dict[str, Any]]:
    """Load questions from the JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['questions']


def run_inference(json_path: str, start_idx: int, end_idx: int, 
                  num_tokens_after_prompt: int = 1, output_dir: str = ".") -> str:
    """Run inference on a range of questions.
    
    Args:
        json_path: Path to the questions JSON file
        start_idx: Starting question index (inclusive)
        end_idx: Ending question index (exclusive)
        num_tokens_after_prompt: Number of tokens to capture after prompt ends
        output_dir: Output directory for results
    """
    
    print("=" * 80)
    print("MISLEADING COGNITIVE INFERENCE - Gemma-3-4B-IT")
    print("=" * 80)
    print(f"Question range: [{start_idx}, {end_idx})")
    print(f"Tokens after prompt: {num_tokens_after_prompt}")
    print("=" * 80)
    
    # Initialize model
    model = GemmaInference(device="auto")
    model.load_model()
    
    # Load questions
    print(f"\n📂 Loading questions from: {json_path}")
    questions = load_questions(json_path)
    print(f"✅ Loaded {len(questions)} questions")
    
    # Validate indices
    if start_idx < 0:
        raise ValueError(f"start_idx must be >= 0, got {start_idx}")
    if end_idx > len(questions):
        print(f"⚠️  Warning: end_idx {end_idx} exceeds dataset size {len(questions)}")
        end_idx = len(questions)
    if start_idx >= end_idx:
        raise ValueError(f"start_idx ({start_idx}) must be < end_idx ({end_idx})")
    
    num_questions = end_idx - start_idx
    print(f"\n📊 Processing {num_questions} questions...")
    
    # Results storage
    results = []
    correct_count = 0
    
    # Process each question
    for q_idx in range(start_idx, end_idx):
        print(f"\n{'='*80}")
        print(f"QUESTION {q_idx} ({q_idx - start_idx + 1}/{num_questions})")
        print("=" * 80)
        
        question = questions[q_idx]
        prompt = question['prompt']
        expected_answer = question['answer']
        num_distractors = question['num_distractors']
        
        print(f"Distractors: {num_distractors}")
        print(f"Expected answer: {expected_answer}")
        print(f"Prompt: {prompt[:100]}...")
        
        # Generate response with hidden states
        try:
            result = model.generate_response_with_hidden_states(prompt, num_tokens_after_prompt)
            response = result['response']
            
            print(f"\nResponse: {response[:200]}...")
            
            # Extract answer
            extracted_answer = extract_answer(response)
            is_correct = extracted_answer == expected_answer
            
            if is_correct:
                correct_count += 1
            
            print(f"\n   Extracted answer: {extracted_answer}")
            print(f"   Expected answer: {expected_answer}")
            print(f"   Correct: {'✓' if is_correct else '✗'}")
            print(f"   Token range: {result['token_range']}")
            
            results.append({
                'question_idx': q_idx,
                'prompt': prompt,
                'expected_answer': expected_answer,
                'extracted_answer': extracted_answer,
                'is_correct': is_correct,
                'num_distractors': num_distractors,
                'raw_response': response,
                'distractors': question.get('distractors', []),
                'prompt_token_count': result['prompt_token_count'],
                'total_token_count': result['total_token_count'],
                'token_range': result['token_range'],
                'residual_stream_hidden_states': result['hidden_states']
            })
            
        except Exception as e:
            print(f"❌ Error: {e}")
            results.append({
                'question_idx': q_idx,
                'error': str(e)
            })
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"misleading_cognitive_inference_q{start_idx}-{end_idx-1}_tokens{num_tokens_after_prompt}_{timestamp}.json"
    output_file = os.path.join(output_dir, filename)
    
    # Prepare output data
    output_data = {
        'metadata': {
            'model_name': model.model_name,
            'inference_date': datetime.now().isoformat(),
            'device': model.device,
            'question_range': {
                'start_idx': start_idx,
                'end_idx': end_idx,
                'num_questions': num_questions
            },
            'source_file': json_path,
            'num_tokens_after_prompt': num_tokens_after_prompt,
            'layers_captured': [18],
            'hidden_size': 2560
        },
        'summary': {
            'total_questions': num_questions,
            'correct': correct_count,
            'incorrect': num_questions - correct_count,
            'accuracy': correct_count / num_questions if num_questions > 0 else 0
        },
        'results': results
    }
    
    # Save to file
    print(f"\n💾 Saving results to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    # Summary
    accuracy = correct_count / num_questions if num_questions > 0 else 0
    
    print("\n" + "=" * 80)
    print("✅ INFERENCE COMPLETE!")
    print("=" * 80)
    print(f"Results saved to: {output_file}")
    print(f"\nSummary:")
    print(f"  - Questions processed: {num_questions}")
    print(f"  - Correct: {correct_count}")
    print(f"  - Incorrect: {num_questions - correct_count}")
    print(f"  - Accuracy: {accuracy*100:.1f}%")
    
    # Breakdown by number of distractors
    print(f"\n  Accuracy by number of distractors:")
    distractor_stats = {}
    for r in results:
        if 'error' not in r:
            n = r['num_distractors']
            if n not in distractor_stats:
                distractor_stats[n] = {'correct': 0, 'total': 0}
            distractor_stats[n]['total'] += 1
            if r['is_correct']:
                distractor_stats[n]['correct'] += 1
    
    for n in sorted(distractor_stats.keys()):
        stats = distractor_stats[n]
        acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        print(f"    {n} distractors: {stats['correct']}/{stats['total']} ({acc*100:.1f}%)")
    
    print("=" * 80)
    
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Run Gemma-3-4B-IT inference on misleading cognitive questions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process questions 0-9, capture 1 token after prompt
    python misleading_cognitive_inference.py misleading_cognitive_2500_20251218_113450.json 0 10 1

    # Process questions 100-199, capture 5 tokens after prompt
    python misleading_cognitive_inference.py misleading_cognitive_2500_20251218_113450.json 100 200 5
        """
    )
    
    parser.add_argument('json_path', type=str,
                        help='Path to misleading cognitive questions JSON file')
    parser.add_argument('start_idx', type=int,
                        help='Starting question index (inclusive)')
    parser.add_argument('end_idx', type=int,
                        help='Ending question index (exclusive)')
    parser.add_argument('num_tokens_after_prompt', type=int,
                        help='Number of tokens to capture after prompt ends')
    parser.add_argument('--output-dir', type=str, default='.',
                        help='Output directory for results (default: current directory)')
    
    args = parser.parse_args()
    
    try:
        run_inference(
            json_path=args.json_path,
            start_idx=args.start_idx,
            end_idx=args.end_idx,
            num_tokens_after_prompt=args.num_tokens_after_prompt,
            output_dir=args.output_dir
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Inference interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

