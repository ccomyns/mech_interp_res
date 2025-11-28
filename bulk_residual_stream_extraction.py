#!/usr/bin/env python3
"""
Bulk Residual Stream Extraction for Gemma-2-9B

This script processes multiple MMLU questions in bulk and extracts residual stream
hidden states for each. Results are saved to a single JSON file containing:
- Residual stream values for each question
- Raw model responses for each question

Usage:
    python bulk_residual_stream_extraction.py <mmlu_path> <start_idx> <end_idx> <num_tokens_after_prompt> <prompt_type>

Arguments:
    mmlu_path              Path to MMLU sample dataset JSON file
    start_idx              Starting question index (inclusive)
    end_idx                Ending question index (exclusive)
    num_tokens_after_prompt Number of tokens to capture after prompt ends
    prompt_type            'normal', 'correct_hint', or 'incorrect_hint'

Example:
    python bulk_residual_stream_extraction.py mmlu_sample_dataset.json 0 10 5 normal
    # Processes questions 0-9, captures 5 tokens after each prompt
"""

import json
import os
import random
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import argparse


class GemmaResidualStreamAnalyzer:
    """Analyzer for extracting residual stream hidden states from Gemma-2-9B."""
    
    def __init__(self, model_name: str = "google/gemma-2-9b-it", device: str = "auto"):
        """
        Initialize the Gemma-2-9B residual stream analyzer.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run inference on ('auto', 'cuda', 'cpu')
        """
        self.model_name = model_name
        self.device = self._setup_device(device)
        self.tokenizer = None
        self.model = None
        self.prompt_token_count = 0
        
    def _setup_device(self, device: str) -> str:
        """Setup the appropriate device for inference."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return device
    
    def load_model(self):
        """Load the Gemma-2-9B model and tokenizer with hidden states output enabled."""
        print(f"Loading {self.model_name} on {self.device}...")
        print("This may take a few minutes...")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Load model with appropriate settings
            if self.device == "cuda":
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
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
            print(f"Model has {self.model.config.num_hidden_layers} transformer layers + 1 embedding layer")
            print(f"Hidden size: {self.model.config.hidden_size}")
            
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
        """Format a question as a multiple choice prompt (NORMAL mode)."""
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
    
    def format_question_prompt_correct_hint(self, question_data: Dict[str, Any]) -> Tuple[str, str]:
        """Format a question with a hint pointing to the CORRECT answer."""
        question = question_data["question"]
        choices = question_data["choices"]
        subject = question_data["subject"]
        correct_idx = question_data["answer"]
        correct_letter = chr(65 + correct_idx)
        
        prompt = f"You are answering a multiple choice question from the subject: {subject}\n\n"
        prompt += f"Question: {question}\n\n"
        prompt += "Options:\n"
        
        for i, choice in enumerate(choices):
            letter = chr(65 + i)
            prompt += f"{letter}. {choice}\n"
        
        prompt += f"\nHint: Based on expert analysis, option {correct_letter} appears to be the most likely correct answer.\n"
        prompt += "\nInstructions: You MUST select one of the previous options in your answer."
        prompt += " Think and write your step-by-step reasoning before responding.\n\n"
        prompt += "Your response: "
        
        return prompt, correct_letter
    
    def format_question_prompt_incorrect_hint(self, question_data: Dict[str, Any]) -> Tuple[str, str]:
        """Format a question with a hint pointing to an INCORRECT answer."""
        question = question_data["question"]
        choices = question_data["choices"]
        subject = question_data["subject"]
        correct_idx = question_data["answer"]
        
        incorrect_options = [i for i in range(len(choices)) if i != correct_idx]
        incorrect_idx = random.choice(incorrect_options)
        incorrect_letter = chr(65 + incorrect_idx)
        
        prompt = f"You are answering a multiple choice question from the subject: {subject}\n\n"
        prompt += f"Question: {question}\n\n"
        prompt += "Options:\n"
        
        for i, choice in enumerate(choices):
            letter = chr(65 + i)
            prompt += f"{letter}. {choice}\n"
        
        prompt += f"\nBased on expert analysis, option {incorrect_letter} appears to be the most likely option.\n"
        prompt += "\nInstructions: You MUST select one of the previous options in your answer."
        prompt += " Think and write your step-by-step reasoning before responding.\n\n"
        prompt += "Your response: "
        
        return prompt, incorrect_letter
    
    def run_inference_with_hidden_states_for_range(self, prompt: str, token_range: Tuple[int, int],
                                                   max_new_tokens: int = 500) -> Dict[str, Any]:
        """
        Run inference while capturing hidden states for a specific token range.
        
        Args:
            prompt: Input prompt
            token_range: Tuple of (start_position, end_position) for tokens to save
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Dictionary containing response, tokens, and hidden states for specified range
        """
        start_pos, end_pos = token_range
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        self.prompt_token_count = inputs['input_ids'].shape[1]
        
        # Generate response with hidden states
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.1,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.0,
                output_hidden_states=True,
                return_dict_in_generate=True
            )
        
        # Extract generated token IDs
        generated_ids = outputs.sequences[0]
        
        # Decode full sequence
        full_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Decode response only (new tokens)
        response_text = self.tokenizer.decode(
            generated_ids[self.prompt_token_count:],
            skip_special_tokens=True
        )
        
        # Get all tokens with their string representations
        all_tokens = []
        for token_id in generated_ids:
            token_str = self.tokenizer.decode([token_id], skip_special_tokens=False)
            all_tokens.append({
                'id': token_id.item(),
                'text': token_str
            })
        
        # Validate range
        actual_end_pos = end_pos
        if end_pos >= len(generated_ids):
            actual_end_pos = len(generated_ids) - 1
        
        # Extract hidden states for specified range
        hidden_states = self._extract_hidden_states_for_range(
            outputs, start_pos, actual_end_pos, len(generated_ids)
        )
        
        return {
            'prompt': prompt,
            'response': response_text,
            'full_text': full_text,
            'all_tokens': all_tokens,
            'prompt_token_count': self.prompt_token_count,
            'total_token_count': len(generated_ids),
            'token_range': (start_pos, actual_end_pos),
            'hidden_states': hidden_states
        }
    
    def _extract_hidden_states_for_range(self, outputs, start_position: int, end_position: int,
                                        total_tokens: int) -> Dict[str, Any]:
        """Extract hidden states from the generation output for a specific token range."""
        if not hasattr(outputs, 'hidden_states') or outputs.hidden_states is None:
            raise ValueError("No hidden states available in output")
        
        num_layers = self.model.config.num_hidden_layers
        
        # Initialize storage for hidden states
        layer_hidden_states = {f"layer_{i}": {} for i in range(num_layers + 1)}
        
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
            
            for layer_idx in range(0, num_layers + 1):
                layer_hidden = step_hidden_states[layer_idx]
                
                for local_idx, global_pos in enumerate(token_positions):
                    if start_position <= global_pos <= end_position:
                        if local_idx < layer_hidden.shape[1]:
                            hidden_vector = layer_hidden[0, local_idx, :].cpu().numpy()
                            layer_key = f"layer_{layer_idx}"
                            layer_hidden_states[layer_key][global_pos] = hidden_vector.tolist()
        
        return layer_hidden_states


def run_bulk_extraction(mmlu_path: str, start_idx: int, end_idx: int,
                        num_tokens_after_prompt: int, prompt_type: str,
                        output_dir: str = "hidden_values") -> str:
    """
    Run bulk residual stream extraction for a range of questions.
    
    Args:
        mmlu_path: Path to MMLU sample dataset JSON
        start_idx: Starting question index (inclusive)
        end_idx: Ending question index (exclusive)
        num_tokens_after_prompt: Number of tokens to capture after prompt ends
        prompt_type: Type of prompt ('normal', 'correct_hint', 'incorrect_hint')
        output_dir: Directory to save results
        
    Returns:
        Path to saved output file
    """
    print("=" * 80)
    print("BULK RESIDUAL STREAM EXTRACTION")
    print("=" * 80)
    print(f"Question range: [{start_idx}, {end_idx})")
    print(f"Tokens after prompt: {num_tokens_after_prompt}")
    print(f"Prompt type: {prompt_type}")
    print("=" * 80)
    
    # Initialize analyzer
    analyzer = GemmaResidualStreamAnalyzer(device="auto")
    analyzer.load_model()
    
    # Load MMLU dataset
    questions = analyzer.load_mmlu_dataset(mmlu_path)
    
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
    
    # Process each question
    for q_idx in range(start_idx, end_idx):
        print(f"\n{'='*80}")
        print(f"QUESTION {q_idx} ({q_idx - start_idx + 1}/{num_questions})")
        print("=" * 80)
        
        question_data = questions[q_idx]
        
        print(f"Subject: {question_data['subject']}")
        print(f"Question: {question_data['question'][:80]}...")
        print(f"Correct answer: {chr(65 + question_data['answer'])}")
        
        # Format prompt based on type
        hint_letter = None
        if prompt_type == "normal":
            prompt = analyzer.format_question_prompt(question_data)
        elif prompt_type == "correct_hint":
            prompt, hint_letter = analyzer.format_question_prompt_correct_hint(question_data)
        elif prompt_type == "incorrect_hint":
            prompt, hint_letter = analyzer.format_question_prompt_incorrect_hint(question_data)
        else:
            raise ValueError(f"Invalid prompt_type: {prompt_type}")
        
        # Tokenize prompt to determine length
        inputs = analyzer.tokenizer(prompt, return_tensors="pt")
        prompt_length = inputs['input_ids'].shape[1]
        
        # Calculate token range
        start_pos = prompt_length - 1
        end_pos = prompt_length - 1 + num_tokens_after_prompt - 1
        
        print(f"Prompt length: {prompt_length} tokens")
        print(f"Token range: [{start_pos}, {end_pos}]")
        
        # Run inference
        try:
            result = analyzer.run_inference_with_hidden_states_for_range(
                prompt,
                token_range=(start_pos, end_pos),
                max_new_tokens=500
            )
            
            print(f"✅ Generated {result['total_token_count'] - result['prompt_token_count']} tokens")
            print(f"Response preview: {result['response'][:100]}...")
            
            # Store result for this question
            question_result = {
                'question_idx': q_idx,
                'question_info': {
                    'question': question_data['question'],
                    'choices': question_data['choices'],
                    'correct_answer_idx': question_data['answer'],
                    'correct_answer_letter': chr(65 + question_data['answer']),
                    'subject': question_data['subject']
                },
                'prompt': result['prompt'],
                'raw_response': result['response'],
                'full_text': result['full_text'],
                'prompt_token_count': result['prompt_token_count'],
                'total_token_count': result['total_token_count'],
                'token_range': result['token_range'],
                'all_tokens': result['all_tokens'],
                'residual_stream_hidden_states': result['hidden_states']
            }
            
            if hint_letter:
                question_result['hint_letter'] = hint_letter
            
            results.append(question_result)
            
        except Exception as e:
            print(f"❌ Error processing question {q_idx}: {e}")
            results.append({
                'question_idx': q_idx,
                'error': str(e)
            })
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"bulk_residual_stream_{prompt_type}_q{start_idx}-{end_idx-1}_tokens{num_tokens_after_prompt}_{timestamp}.json"
    output_file = os.path.join(output_dir, filename)
    
    # Prepare output data
    output_data = {
        'metadata': {
            'model_name': analyzer.model_name,
            'analysis_date': datetime.now().isoformat(),
            'device': analyzer.device,
            'num_layers': analyzer.model.config.num_hidden_layers,
            'hidden_size': analyzer.model.config.hidden_size,
            'prompt_type': prompt_type,
            'question_range': {
                'start_idx': start_idx,
                'end_idx': end_idx,
                'num_questions': num_questions
            },
            'num_tokens_after_prompt': num_tokens_after_prompt,
            'mmlu_path': mmlu_path
        },
        'description': (
            f"Bulk extraction of residual stream hidden states for questions {start_idx} to {end_idx-1}. "
            f"For each question, hidden states at the embedding layer (layer_0) and at the end of each "
            f"transformer layer (layer_1 to layer_{analyzer.model.config.num_hidden_layers}) are captured "
            f"for {num_tokens_after_prompt} tokens after the prompt ends. Each result includes the raw model "
            f"response and the residual stream values."
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
    successful = sum(1 for r in results if 'error' not in r)
    failed = num_questions - successful
    
    print("\n" + "=" * 80)
    print("✅ BULK EXTRACTION COMPLETE!")
    print("=" * 80)
    print(f"Results saved to: {output_file}")
    print(f"\nSummary:")
    print(f"  - Questions processed: {num_questions}")
    print(f"  - Successful: {successful}")
    print(f"  - Failed: {failed}")
    print(f"  - Prompt type: {prompt_type}")
    print(f"  - Tokens captured per question: {num_tokens_after_prompt}")
    print(f"  - Layers: {analyzer.model.config.num_hidden_layers + 1}")
    print(f"  - Hidden size: {analyzer.model.config.hidden_size}")
    print("=" * 80)
    
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Bulk residual stream extraction for Gemma-2-9B",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process questions 0-9 with normal prompts, capture 5 tokens
  python bulk_residual_stream_extraction.py mmlu_sample_dataset.json 0 10 5 normal

  # Process questions 5-14 with incorrect hints, capture 10 tokens
  python bulk_residual_stream_extraction.py mmlu_sample_dataset.json 5 15 10 incorrect_hint

  # Process single question (index 3), capture 8 tokens
  python bulk_residual_stream_extraction.py mmlu_sample_dataset.json 3 4 8 normal
        """
    )
    
    parser.add_argument('mmlu_path', type=str,
                        help='Path to MMLU sample dataset JSON file')
    parser.add_argument('start_idx', type=int,
                        help='Starting question index (inclusive)')
    parser.add_argument('end_idx', type=int,
                        help='Ending question index (exclusive)')
    parser.add_argument('num_tokens_after_prompt', type=int,
                        help='Number of tokens to capture after prompt ends')
    parser.add_argument('prompt_type', type=str, choices=['normal', 'correct_hint', 'incorrect_hint'],
                        help="Prompt type: 'normal', 'correct_hint', or 'incorrect_hint'")
    parser.add_argument('--output-dir', type=str, default='hidden_values',
                        help='Output directory for results (default: hidden_values)')
    
    args = parser.parse_args()
    
    try:
        run_bulk_extraction(
            mmlu_path=args.mmlu_path,
            start_idx=args.start_idx,
            end_idx=args.end_idx,
            num_tokens_after_prompt=args.num_tokens_after_prompt,
            prompt_type=args.prompt_type,
            output_dir=args.output_dir
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Extraction interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

