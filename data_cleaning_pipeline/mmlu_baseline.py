#!/usr/bin/env python3
"""
MMLU Baseline Inference for Gemma-3-4B-IT

This script runs inference on MMLU questions with a normal, unbiased prompt
(no hints) to establish baseline model performance.

Usage:
    python mmlu_baseline.py <mmlu_path> <start_idx> <end_idx>

Arguments:
    mmlu_path   Path to MMLU sample dataset JSON file
    start_idx   Starting question index (inclusive)
    end_idx     Ending question index (exclusive)

Example:
    python mmlu_baseline.py mmlu_sample_dataset.json 0 100
    # Processes questions 0-99
"""

import json
import os
import re
from typing import Dict, List, Any, Optional
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse


class GemmaBaselineInference:
    """Run baseline inference on MMLU questions using Gemma-3-4B-IT."""
    
    def __init__(self, model_name: str = "google/gemma-3-4b-it", device: str = "auto"):
        """
        Initialize the Gemma-3-4B inference runner.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run inference on ('auto', 'cuda', 'cpu')
        """
        self.model_name = model_name
        self.device = self._setup_device(device)
        self.tokenizer = None
        self.model = None
        
    def _setup_device(self, device: str) -> str:
        """Setup the appropriate device for inference."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return device
    
    def load_model(self):
        """Load the Gemma-3-4B model and tokenizer."""
        print(f"Loading {self.model_name} on {self.device}...")
        print("This may take a few minutes...")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Load model (no need for output_hidden_states for baseline)
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
            
            print(f"✅ Model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def load_mmlu_dataset(self, json_path: str) -> List[Dict[str, Any]]:
        """Load questions from MMLU sample dataset JSON."""
        print(f"\n📂 Loading MMLU dataset from: {json_path}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        questions = data['questions']
        print(f"✅ Loaded {len(questions)} questions")
        
        return questions
    
    def format_question_prompt(self, question_data: Dict[str, Any]) -> str:
        """Format a question as a multiple choice prompt (normal/unbiased mode)."""
        question = question_data["question"]
        choices = question_data["choices"]
        subject = question_data["subject"]
        
        prompt = f"You are answering a multiple choice question from the subject: {subject}\n\n"
        prompt += f"Question: {question}\n\n"
        prompt += "Options:\n"
        
        for i, choice in enumerate(choices):
            letter = chr(65 + i)
            prompt += f"{letter}. {choice}\n"
        
        prompt += "\nInstructions: You MUST select one of the previous options in your answer."
        prompt += " Think and write your step-by-step reasoning before responding.\n\n"
        prompt += "Your response: "
        
        return prompt
    
    def run_inference(self, prompt: str, max_new_tokens: int = 1500) -> str:
        """
        Run inference and return the model's response.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Model's response text
        """
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_token_count = inputs['input_ids'].shape[1]
        
        # Generate response
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
        
        # Decode response only (new tokens)
        response_text = self.tokenizer.decode(
            outputs[0][prompt_token_count:],
            skip_special_tokens=True
        )
        
        return response_text
    
    def extract_answer(self, response: str) -> str:
        """
        Extract the answer letter from the model's response.
        
        Returns:
            Single letter (A, B, C, or D) or 'N.A.' if not found
        """
        # Primary patterns - look for explicit answer statements
        primary_patterns = [
            r'\*\*Final Answer:\*\*\s*\**\s*([A-D])\b',
            r'\*\*Answer:\*\*\s*\**\s*([A-D])\b',
            r'Final Answer:\s*\**\s*([A-D])\b',
            r'(?:The|My) answer is[:\s]+\**([A-D])\b',
            r'(?:correct|best) (?:answer|option) is[:\s]+\**([A-D])\b',
            r'I (?:would )?(?:choose|select|pick)[:\s]+\**([A-D])\b',
            r'\bAnswer:\s*\**([A-D])\b',
        ]
        
        for pattern in primary_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        
        # Check last 100 characters for common answer patterns
        last_100 = response[-100:] if len(response) > 100 else response
        
        additional_patterns = [
            r'Therefore,?\s*(?:my|the)\s*answer\s*is\s*\**([A-D])\b',
            r'\*\*Final Answer:\*\*\s*([A-D])\b',
            r'\*\*Answer:\*\*\s*([A-D])\b',
            r'Answer:\s*([A-D])\b',
        ]
        
        for pattern in additional_patterns:
            match = re.search(pattern, last_100, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        
        # Fallback: look for standalone letter at the end
        end_pattern = r'\b([A-D])\s*\.?\s*$'
        match = re.search(end_pattern, response.strip())
        if match:
            return match.group(1).upper()
        
        return "N.A."


def run_baseline_inference(mmlu_path: str,
                           start_idx: int,
                           end_idx: int,
                           output_dir: str = "baseline_results",
                           max_new_tokens: int = 1500) -> str:
    """
    Run baseline inference on a range of MMLU questions.
    
    Args:
        mmlu_path: Path to MMLU dataset JSON
        start_idx: Starting question index (inclusive)
        end_idx: Ending question index (exclusive)
        output_dir: Directory to save results
        max_new_tokens: Maximum tokens to generate per question
        
    Returns:
        Path to the output file
    """
    print("=" * 80)
    print("MMLU BASELINE INFERENCE")
    print("=" * 80)
    print(f"Question range: {start_idx} to {end_idx - 1}")
    print()
    
    # Initialize inference runner
    runner = GemmaBaselineInference()
    runner.load_model()
    
    # Load dataset
    questions = runner.load_mmlu_dataset(mmlu_path)
    
    # Validate indices
    num_questions = end_idx - start_idx
    if start_idx < 0 or end_idx > len(questions):
        raise ValueError(f"Invalid question range. Dataset has {len(questions)} questions.")
    
    print(f"\n📊 Processing {num_questions} questions...")
    print("=" * 80)
    
    results = []
    correct_count = 0
    
    for i, q_idx in enumerate(range(start_idx, end_idx)):
        question_data = questions[q_idx]
        
        print(f"\n[{i + 1}/{num_questions}] Question {q_idx}: {question_data['subject']}")
        
        try:
            # Format prompt (normal, unbiased)
            prompt = runner.format_question_prompt(question_data)
            
            # Run inference
            response = runner.run_inference(prompt, max_new_tokens)
            
            # Extract answer
            extracted_answer = runner.extract_answer(response)
            
            # Get correct answer letter
            correct_idx = question_data['answer']
            correct_letter = chr(65 + correct_idx)
            
            # Check if correct
            is_correct = (extracted_answer == correct_letter)
            if is_correct:
                correct_count += 1
            
            # Determine if response exceeded token budget
            exceeded_budget = len(response) >= max_new_tokens * 3  # Rough estimate
            
            print(f"   Correct: {correct_letter} | Model: {extracted_answer} | {'✅' if is_correct else '❌'}")
            
            # Build result
            question_result = {
                'question_idx': q_idx,
                'subject': question_data['subject'],
                'question': question_data['question'],
                'choices': question_data['choices'],
                'correct_answer_idx': correct_idx,
                'correct_answer_letter': correct_letter,
                'raw_response': response,
                'full_text': prompt + response,
                'extracted_answer': extracted_answer,
                'is_correct': is_correct,
                'exceeded_token_budget': exceeded_budget
            }
            
            results.append(question_result)
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results.append({
                'question_idx': q_idx,
                'subject': question_data.get('subject', 'unknown'),
                'error': str(e)
            })
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"baseline_q{start_idx}-{end_idx - 1}_{timestamp}.json"
    output_file = os.path.join(output_dir, filename)
    
    # Calculate statistics
    successful = sum(1 for r in results if 'error' not in r)
    accuracy = (correct_count / successful * 100) if successful > 0 else 0
    
    # Prepare output data
    output_data = {
        'metadata': {
            'model_name': runner.model_name,
            'inference_date': datetime.now().isoformat(),
            'device': runner.device,
            'prompt_type': 'normal_unbiased',
            'question_range': {
                'start_idx': start_idx,
                'end_idx': end_idx,
                'num_questions': num_questions
            },
            'max_new_tokens': max_new_tokens,
            'mmlu_path': mmlu_path
        },
        'statistics': {
            'total_questions': num_questions,
            'successful': successful,
            'failed': num_questions - successful,
            'correct': correct_count,
            'incorrect': successful - correct_count,
            'accuracy_percent': round(accuracy, 2),
            'na_count': sum(1 for r in results if r.get('extracted_answer') == 'N.A.')
        },
        'description': (
            f"Baseline inference results for questions {start_idx} to {end_idx - 1}. "
            f"Model: {runner.model_name}. Prompt type: normal (no hints). "
            f"This establishes baseline performance to compare against hint experiments."
        ),
        'results': results
    }
    
    # Save to file
    print(f"\n💾 Saving results to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    file_size = os.path.getsize(output_file) / (1024 * 1024)
    print(f"   File size: {file_size:.2f} MB")
    
    # Summary
    print("\n" + "=" * 80)
    print("✅ BASELINE INFERENCE COMPLETE!")
    print("=" * 80)
    print(f"Results saved to: {output_file}")
    print(f"\nSummary:")
    print(f"  - Questions processed: {num_questions}")
    print(f"  - Successful: {successful}")
    print(f"  - Failed: {num_questions - successful}")
    print(f"  - Correct: {correct_count}")
    print(f"  - Incorrect: {successful - correct_count}")
    print(f"  - Accuracy: {accuracy:.2f}%")
    print(f"  - N.A. (couldn't extract): {output_data['statistics']['na_count']}")
    print("=" * 80)
    
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Run baseline MMLU inference with Gemma-3-4B-IT (no hints)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process questions 0-99
  python mmlu_baseline.py mmlu_sample_dataset.json 0 100

  # Process questions 100-199
  python mmlu_baseline.py mmlu_sample_dataset.json 100 200

  # Process all questions (0-1139)
  python mmlu_baseline.py mmlu_sample_dataset.json 0 1140
        """
    )
    
    parser.add_argument('mmlu_path', type=str,
                        help='Path to MMLU sample dataset JSON file')
    parser.add_argument('start_idx', type=int,
                        help='Starting question index (inclusive)')
    parser.add_argument('end_idx', type=int,
                        help='Ending question index (exclusive)')
    parser.add_argument('--output-dir', type=str, default='baseline_results',
                        help='Output directory for results (default: baseline_results)')
    parser.add_argument('--max-tokens', type=int, default=1500,
                        help='Maximum new tokens to generate per question (default: 1500)')
    
    args = parser.parse_args()
    
    try:
        run_baseline_inference(
            mmlu_path=args.mmlu_path,
            start_idx=args.start_idx,
            end_idx=args.end_idx,
            output_dir=args.output_dir,
            max_new_tokens=args.max_tokens
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Inference interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

