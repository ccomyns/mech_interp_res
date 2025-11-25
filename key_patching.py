#!/usr/bin/env python3
"""
Gemma-2-9B Key Activation Patching

This script performs activation patching experiments by:
1. Loading cached key values from a JSON file (from key_extraction.py)
2. Running inference with an incorrect hint prompt
3. Patching the keys at a specific layer with the cached keys
4. Comparing the output with and without patching

This allows us to test whether keys from one condition (e.g., normal prompt)
can influence the model's behavior in another condition (e.g., incorrect hint).
"""

import json
import os
import random
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np


class GemmaKeyPatcher:
    def __init__(self, model_name: str = "google/gemma-2-9b-it", device: str = "auto"):
        """
        Initialize the Gemma-2-9B key patcher.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run inference on ('auto', 'cuda', 'cpu')
        """
        self.model_name = model_name
        self.device = self._setup_device(device)
        self.tokenizer = None
        self.model = None
        self.cached_keys = None
        self.cached_metadata = None
        
    def _setup_device(self, device: str) -> str:
        """Setup the appropriate device for inference."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return device
    
    def load_model(self):
        """Load the Gemma-2-9B model and tokenizer."""
        print(f"Loading {self.model_name} on {self.device}...")
        print("This may take a few minutes...")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Load model
            if self.device == "cuda":
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
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
            print(f"Model has {self.model.config.num_hidden_layers} transformer layers")
            
            # If keys were already loaded, convert them to model dtype and device
            if hasattr(self, 'key_tensor') and self.key_tensor is not None:
                print(f"\n🔄 Converting cached keys to model dtype and device...")
                self.key_tensor = self.key_tensor.to(self.device).to(self.model.dtype)
                print(f"   ✅ Keys converted to {self.model.dtype} on {self.device}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def load_cached_keys(self, json_path: str) -> Dict[str, Any]:
        """
        Load cached key values from a JSON file.
        
        Args:
            json_path: Path to the cached keys JSON file
            
        Returns:
            Dictionary with loaded data
        """
        print(f"\n📂 Loading cached keys from: {json_path}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.cached_metadata = data['metadata']
        self.cached_keys = data['keys']
        
        print(f"✅ Cached keys loaded successfully")
        print(f"   Source prompt type: {self.cached_metadata['prompt_type']}")
        print(f"   Target layer: {self.cached_metadata['target_layer']}")
        print(f"   Token range: {self.cached_metadata['token_range']}")
        print(f"   Number of KV heads: {self.cached_metadata['num_key_value_heads']}")
        print(f"   Head dimension: {self.cached_metadata['head_dim']}")
        
        # Convert keys to tensors
        self._prepare_key_tensors()
        
        # If model was already loaded, convert to model dtype and device
        if self.model is not None:
            print(f"\n🔄 Converting cached keys to model dtype and device...")
            self.key_tensor = self.key_tensor.to(self.device).to(self.model.dtype)
            print(f"   ✅ Keys converted to {self.model.dtype} on {self.device}")
        
        return data
    
    def _prepare_key_tensors(self):
        """Convert loaded keys from lists to PyTorch tensors."""
        print(f"\n🔄 Converting keys to tensors...")
        
        # Keys are organized as: cached_keys[token_pos]["kv_head_X"] = [list]
        # We need to convert to tensor format: [num_kv_heads, seq_len, head_dim]
        
        token_positions = sorted([int(pos) for pos in self.cached_keys.keys()])
        num_kv_heads = self.cached_metadata['num_key_value_heads']
        seq_len = len(token_positions)
        
        # Get actual head_dim from the first key vector (more reliable than metadata)
        first_token_pos = str(token_positions[0])
        first_key_vector = self.cached_keys[first_token_pos]["kv_head_0"]
        actual_head_dim = len(first_key_vector)
        
        metadata_head_dim = self.cached_metadata['head_dim']
        if actual_head_dim != metadata_head_dim:
            print(f"   ⚠️  Note: Metadata head_dim={metadata_head_dim}, but actual vectors are {actual_head_dim}")
            print(f"   Using actual dimension: {actual_head_dim}")
        
        head_dim = actual_head_dim
        
        # Create tensor: [num_kv_heads, seq_len, head_dim]
        key_tensor = torch.zeros(num_kv_heads, seq_len, head_dim)
        
        for local_idx, token_pos in enumerate(token_positions):
            for head_idx in range(num_kv_heads):
                key_vector = self.cached_keys[str(token_pos)][f"kv_head_{head_idx}"]
                key_tensor[head_idx, local_idx, :] = torch.tensor(key_vector)
        
        # Store tensor and position mapping
        # Keep as float32 for now, will convert to model dtype when model is loaded
        self.key_tensor = key_tensor
        self.actual_head_dim = head_dim
        self.token_position_map = {local_idx: token_pos for local_idx, token_pos in enumerate(token_positions)}
        self.global_to_local = {token_pos: local_idx for local_idx, token_pos in enumerate(token_positions)}
        
        print(f"   ✅ Converted keys tensor shape: {self.key_tensor.shape}")
        print(f"   Token positions: {token_positions}")
        print(f"   Head dimension: {head_dim}")
        print(f"   Note: Will convert to model dtype after model is loaded")
    
    def load_mmlu_dataset(self, json_path: str) -> List[Dict[str, Any]]:
        """Load questions from MMLU sample dataset JSON."""
        print(f"\n📂 Loading MMLU dataset from: {json_path}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        questions = data['questions']
        print(f"✅ Loaded {len(questions)} questions")
        
        return questions
    
    def format_question_prompt_incorrect_hint(self, question_data: Dict[str, Any]) -> Tuple[str, str, str]:
        """
        Format a question with a hint pointing to an INCORRECT answer.
        
        Args:
            question_data: Question dictionary
            
        Returns:
            Tuple of (prompt, incorrect_letter, correct_letter)
        """
        question = question_data["question"]
        choices = question_data["choices"]
        subject = question_data["subject"]
        correct_idx = question_data["answer"]
        correct_letter = chr(65 + correct_idx)
        
        # Select a random incorrect answer
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
        
        return prompt, incorrect_letter, correct_letter
    
    def run_inference_without_patching(self, prompt: str, max_new_tokens: int = 500) -> Dict[str, Any]:
        """
        Run inference WITHOUT patching (baseline).
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Dictionary with response and tokens
        """
        print("\n🤖 Running baseline inference (NO patching)...")
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_token_count = inputs['input_ids'].shape[1]
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.1,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.0,
                return_dict_in_generate=True
            )
        
        generated_ids = outputs.sequences[0]
        response_text = self.tokenizer.decode(
            generated_ids[prompt_token_count:],
            skip_special_tokens=True
        )
        
        print(f"   ✅ Generated {len(generated_ids) - prompt_token_count} tokens")
        
        return {
            'response': response_text,
            'full_sequence': generated_ids,
            'prompt_token_count': prompt_token_count
        }
    
    def run_inference_with_patching(self, prompt: str, patch_offset: int = 0, 
                                   max_new_tokens: int = 500) -> Dict[str, Any]:
        """
        Run inference WITH key patching.
        
        Args:
            prompt: Input prompt
            patch_offset: Offset to apply to cached key positions. 
                         Use 'last_prompt_token' to patch at the last token of the target prompt,
                         or an integer offset from the last prompt token.
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Dictionary with response and tokens
        """
        print("\n🔧 Running inference WITH key patching...")
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_token_count = inputs['input_ids'].shape[1]
        
        target_layer = self.cached_metadata['target_layer']
        token_range = self.cached_metadata['token_range']
        cached_start_pos, cached_end_pos = token_range
        
        # Calculate the offset to apply to cached positions
        # patch_offset determines where to patch relative to the target prompt
        target_start_pos = cached_start_pos + patch_offset
        target_end_pos = cached_end_pos + patch_offset
        
        print(f"   Patching layer: {target_layer}")
        print(f"   Cached key positions (from normal prompt): {cached_start_pos} to {cached_end_pos}")
        print(f"   Target prompt length: {prompt_token_count} tokens")
        print(f"   Patch offset: {patch_offset}")
        print(f"   Target patching positions: {target_start_pos} to {target_end_pos}")
        
        # Create a mapping from target positions to cached key indices
        # The cached keys are at positions cached_start_pos to cached_end_pos
        # We want to patch them into positions target_start_pos to target_end_pos
        position_mapping = {}
        for i, cached_pos in enumerate(range(cached_start_pos, cached_end_pos + 1)):
            target_pos = target_start_pos + i
            if cached_pos in self.global_to_local:
                local_idx = self.global_to_local[cached_pos]
                position_mapping[target_pos] = local_idx
        
        print(f"   Position mapping: {len(position_mapping)} positions will be patched")
        
        # Create a hook to patch keys during generation
        def patch_keys_hook(module, args, kwargs, output):
            """Hook to replace keys with cached keys."""
            # output is typically (attn_output, attn_weights, past_key_value)
            if isinstance(output, tuple) and len(output) >= 3:
                past_key_value = output[2]
                if past_key_value is not None and isinstance(past_key_value, tuple):
                    keys, values = past_key_value
                    # keys shape: [batch, num_kv_heads, seq_len, head_dim]
                    
                    batch_size, num_kv_heads, seq_len, head_dim = keys.shape
                    
                    # Verify dimensions match
                    if head_dim != self.key_tensor.shape[2]:
                        print(f"\n⚠️  Warning: Head dimension mismatch!")
                        print(f"   Model keys: {head_dim}, Cached keys: {self.key_tensor.shape[2]}")
                        print(f"   Skipping patching for this forward pass")
                        return output
                    
                    # Patch keys at specified target positions using cached keys
                    for target_pos, local_idx in position_mapping.items():
                        if target_pos < seq_len:
                            # Replace keys at target position with cached keys
                            keys[0, :, target_pos, :] = self.key_tensor[:, local_idx, :]
                    
                    # Return modified output
                    return (output[0], output[1], (keys, values))
            
            return output
        
        # Register hook
        layer = self.model.model.layers[target_layer]
        hook_handle = layer.self_attn.register_forward_hook(patch_keys_hook, with_kwargs=True)
        
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.1,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.0,
                    return_dict_in_generate=True
                )
        finally:
            hook_handle.remove()
        
        generated_ids = outputs.sequences[0]
        response_text = self.tokenizer.decode(
            generated_ids[prompt_token_count:],
            skip_special_tokens=True
        )
        
        print(f"   ✅ Generated {len(generated_ids) - prompt_token_count} tokens with patching")
        
        return {
            'response': response_text,
            'full_sequence': generated_ids,
            'prompt_token_count': prompt_token_count
        }
    
    def compare_results(self, baseline: Dict[str, Any], patched: Dict[str, Any],
                       question_data: Dict[str, Any], incorrect_letter: str) -> Dict[str, Any]:
        """
        Compare baseline and patched results.
        
        Args:
            baseline: Baseline inference results
            patched: Patched inference results
            question_data: Question data
            incorrect_letter: The incorrect hint letter
            
        Returns:
            Comparison dictionary
        """
        print("\n" + "=" * 80)
        print("COMPARISON: Baseline vs Patched")
        print("=" * 80)
        
        correct_letter = chr(65 + question_data['answer'])
        
        print(f"\nQuestion: {question_data['question'][:80]}...")
        print(f"Correct answer: {correct_letter}")
        print(f"Incorrect hint: {incorrect_letter}")
        
        print("\n" + "-" * 80)
        print("BASELINE (No Patching):")
        print("-" * 80)
        print(baseline['response'][:500] + ("..." if len(baseline['response']) > 500 else ""))
        
        print("\n" + "-" * 80)
        print("PATCHED (With Key Patching):")
        print("-" * 80)
        print(patched['response'][:500] + ("..." if len(patched['response']) > 500 else ""))
        
        # Check which answer was selected
        def extract_answer(response: str) -> Optional[str]:
            """Try to extract the selected answer letter."""
            response_upper = response.upper()
            for letter in ['A', 'B', 'C', 'D']:
                # Look for patterns like "**A." or "A)" or "Answer: A"
                if f"**{letter}." in response_upper or f"**{letter} " in response_upper:
                    return letter
                if response_upper.startswith(letter):
                    return letter
            return None
        
        baseline_answer = extract_answer(baseline['response'])
        patched_answer = extract_answer(patched['response'])
        
        print("\n" + "=" * 80)
        print("SUMMARY:")
        print("=" * 80)
        print(f"Correct answer: {correct_letter}")
        print(f"Incorrect hint: {incorrect_letter}")
        print(f"Baseline selected: {baseline_answer if baseline_answer else 'Unknown'}")
        print(f"Patched selected: {patched_answer if patched_answer else 'Unknown'}")
        
        if baseline_answer == incorrect_letter and patched_answer == correct_letter:
            print("\n✅ SUCCESS: Patching changed incorrect answer to correct!")
        elif baseline_answer == incorrect_letter and patched_answer != incorrect_letter:
            print(f"\n⚠️  PARTIAL: Patching changed answer from {incorrect_letter} to {patched_answer}")
        elif baseline_answer != patched_answer:
            print(f"\n📊 CHANGE: Patching changed answer from {baseline_answer} to {patched_answer}")
        else:
            print(f"\n❌ NO CHANGE: Both selected {baseline_answer}")
        
        print("=" * 80)
        
        return {
            'correct_letter': correct_letter,
            'incorrect_hint': incorrect_letter,
            'baseline_answer': baseline_answer,
            'patched_answer': patched_answer,
            'baseline_response': baseline['response'],
            'patched_response': patched['response']
        }
    
    def save_results(self, comparison: Dict[str, Any], question_data: Dict[str, Any],
                    prompt: str, patch_offset: int, output_dir: str = "patching_results") -> str:
        """Save patching experiment results."""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"key_patching_q{question_data.get('id', 'unknown')}_layer{self.cached_metadata['target_layer']}_{timestamp}.json"
        output_file = os.path.join(output_dir, filename)
        
        cached_token_range = self.cached_metadata['token_range']
        target_token_range = [cached_token_range[0] + patch_offset, cached_token_range[1] + patch_offset]
        
        output_data = {
            'metadata': {
                'experiment_date': datetime.now().isoformat(),
                'model_name': self.model_name,
                'device': self.device,
                'patching_config': {
                    'source_prompt_type': self.cached_metadata['prompt_type'],
                    'target_prompt_type': 'incorrect_hint',
                    'target_layer': self.cached_metadata['target_layer'],
                    'cached_token_range': cached_token_range,
                    'patch_offset': patch_offset,
                    'target_token_range': target_token_range
                }
            },
            'question_info': {
                'question': question_data['question'],
                'choices': question_data['choices'],
                'correct_answer_idx': question_data['answer'],
                'correct_answer_letter': comparison['correct_letter'],
                'subject': question_data['subject']
            },
            'prompt': prompt,
            'results': {
                'incorrect_hint': comparison['incorrect_hint'],
                'baseline_answer': comparison['baseline_answer'],
                'patched_answer': comparison['patched_answer'],
                'baseline_response': comparison['baseline_response'],
                'patched_response': comparison['patched_response']
            }
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results saved to: {output_file}")
        return output_file


def main():
    """Main function with CLI interface."""
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] in ["--help", "-h"]:
            print("\n" + "=" * 80)
            print("GEMMA-2-9B KEY ACTIVATION PATCHING")
            print("=" * 80)
            print("\nUsage:")
            print("  python key_patching.py <cached_keys_json> <mmlu_path> <question_idx> <patch_position> [max_tokens]")
            print("\nArguments:")
            print("  cached_keys_json   Path to cached keys JSON file (from key_extraction.py)")
            print("  mmlu_path          Path to MMLU sample dataset JSON file")
            print("  question_idx       Question index (0-based)")
            print("  patch_position     Where to patch relative to incorrect hint prompt:")
            print("                     - 'last_prompt_token': Patch at last token of incorrect hint prompt")
            print("                     - Integer N: Patch starting at position N in incorrect hint prompt")
            print("  max_tokens         (optional) Maximum number of tokens to generate [default: 500]")
            print("\nDescription:")
            print("  Performs activation patching experiment by:")
            print("  1. Loading cached key values from one condition (e.g., normal prompt)")
            print("  2. Running inference with incorrect hint prompt (baseline)")
            print("  3. Running inference with incorrect hint prompt + patched keys")
            print("  4. Comparing the outputs to see if patching changes the answer")
            print("\nKey Position Mapping:")
            print("  The cached keys were extracted at specific positions in the normal prompt.")
            print("  This script maps those keys to new positions in the incorrect hint prompt.")
            print("  Example: If cached keys are at positions [114, 115, 116] (normal prompt)")
            print("           and incorrect hint prompt is 130 tokens, using 'last_prompt_token'")
            print("           will patch cached keys into positions [129, 130, 131]")
            print("\nExamples:")
            print("  # Patch at last token of incorrect hint prompt")
            print("  python key_patching.py keys_normal_layer1_range_114_118.json mmlu.json 0 last_prompt_token")
            print()
            print("  # Patch starting at specific position in incorrect hint prompt")
            print("  python key_patching.py keys_normal_layer1_range_114_118.json mmlu.json 0 129")
            print()
            print("  # Same with longer generation")
            print("  python key_patching.py keys_normal_layer1_range_114_118.json mmlu.json 0 last_prompt_token 600")
            print("=" * 80)
            sys.exit(0)
        
        if len(sys.argv) < 5:
            print("❌ Error: Insufficient arguments")
            print("Run 'python key_patching.py --help' for usage information")
            sys.exit(1)
        
        cached_keys_path = sys.argv[1]
        mmlu_path = sys.argv[2]
        question_idx = int(sys.argv[3])
        patch_position_arg = sys.argv[4]
        max_tokens = int(sys.argv[5]) if len(sys.argv) > 5 else 500
    else:
        print("❌ Error: No arguments provided")
        print("Run 'python key_patching.py --help' for usage information")
        sys.exit(1)
    
    # Run the patching experiment
    try:
        print("=" * 80)
        print("GEMMA-2-9B KEY ACTIVATION PATCHING EXPERIMENT")
        print("=" * 80)
        
        # Initialize patcher
        patcher = GemmaKeyPatcher(device="auto")
        
        # Load cached keys
        patcher.load_cached_keys(cached_keys_path)
        
        # Load model
        patcher.load_model()
        
        # Load MMLU dataset
        questions = patcher.load_mmlu_dataset(mmlu_path)
        
        # Validate question index
        if question_idx >= len(questions):
            print(f"❌ Error: Question index {question_idx} out of range (0-{len(questions)-1})")
            sys.exit(1)
        
        question_data = questions[question_idx]
        
        print(f"\n📝 Selected question {question_idx}:")
        print(f"   Subject: {question_data['subject']}")
        print(f"   Question: {question_data['question'][:80]}...")
        print(f"   Correct answer: {chr(65 + question_data['answer'])}")
        
        # Format incorrect hint prompt
        prompt, incorrect_letter, correct_letter = patcher.format_question_prompt_incorrect_hint(question_data)
        
        # Tokenize to get incorrect hint prompt length
        incorrect_hint_inputs = patcher.tokenizer(prompt, return_tensors="pt")
        incorrect_hint_prompt_length = incorrect_hint_inputs['input_ids'].shape[1]
        
        # Calculate patch_offset based on patch_position_arg
        cached_start_pos = patcher.cached_metadata['token_range'][0]
        
        if patch_position_arg.lower() == 'last_prompt_token':
            # Patch at the last token of the incorrect hint prompt
            # Offset = (incorrect_hint_last_token_pos) - (cached_start_pos)
            patch_offset = (incorrect_hint_prompt_length - 1) - cached_start_pos
            print(f"\n📍 Patch position: last_prompt_token")
            print(f"   Incorrect hint prompt length: {incorrect_hint_prompt_length} tokens")
            print(f"   Last prompt token position: {incorrect_hint_prompt_length - 1}")
            print(f"   Cached keys start at: {cached_start_pos}")
            print(f"   Calculated offset: {patch_offset}")
        else:
            # User specified an absolute position
            try:
                target_start_position = int(patch_position_arg)
                patch_offset = target_start_position - cached_start_pos
                print(f"\n📍 Patch position: {target_start_position}")
                print(f"   Cached keys start at: {cached_start_pos}")
                print(f"   Calculated offset: {patch_offset}")
            except ValueError:
                print(f"❌ Error: Invalid patch_position '{patch_position_arg}'")
                print("   Must be 'last_prompt_token' or an integer")
                sys.exit(1)
        
        print(f"\n🎯 Experiment setup:")
        print(f"   Source keys: {patcher.cached_metadata['prompt_type']} prompt")
        print(f"   Target prompt: incorrect_hint (hint: {incorrect_letter})")
        print(f"   Correct answer: {correct_letter}")
        print(f"   Layer to patch: {patcher.cached_metadata['target_layer']}")
        print(f"   Cached token range: {patcher.cached_metadata['token_range']}")
        print(f"   Patch offset: {patch_offset}")
        
        # Run baseline inference (no patching)
        baseline = patcher.run_inference_without_patching(prompt, max_tokens)
        
        # Run patched inference
        patched = patcher.run_inference_with_patching(prompt, patch_offset, max_tokens)
        
        # Compare results
        comparison = patcher.compare_results(baseline, patched, question_data, incorrect_letter)
        
        # Save results
        patcher.save_results(comparison, question_data, prompt, patch_offset)
        
        print("\n✅ Experiment complete!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Experiment interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

