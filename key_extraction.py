#!/usr/bin/env python3
"""
Gemma-2-9B Key Value Extraction for Activation Patching

This script captures attention key values at a specific layer for tokens after the prompt.
The pipeline:
1. Select a question from MMLU dataset by index
2. Format the question with normal/correct_hint/incorrect_hint
3. Accept CLI input to specify layer and number of tokens
4. Run inference and capture key values from specified layer
5. Save key values to JSON for later use in activation patching
"""

import json
import os
import random
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np


class GemmaKeyValueExtractor:
    def __init__(self, model_name: str = "google/gemma-2-9b-it", device: str = "auto"):
        """
        Initialize the Gemma-2-9B key value extractor.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run inference on ('auto', 'cuda', 'cpu')
        """
        self.model_name = model_name
        self.device = self._setup_device(device)
        self.tokenizer = None
        self.model = None
        self.captured_keys = None
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
        """Load the Gemma-2-9B model and tokenizer."""
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
                    output_attentions=True  # Enable attention output
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                    output_attentions=True  # Enable attention output
                ).to(self.device)
            
            print(f"✅ Model loaded successfully on {self.device}")
            print(f"Model has {self.model.config.num_hidden_layers} transformer layers")
            print(f"Hidden size: {self.model.config.hidden_size}")
            print(f"\nAttention Architecture (Grouped-Query Attention):")
            print(f"  - Query heads: {self.model.config.num_attention_heads}")
            print(f"  - Key-Value heads: {self.model.config.num_key_value_heads}")
            print(f"  - Head dimension: {self.model.config.hidden_size // self.model.config.num_attention_heads}")
            print(f"  - Each KV head is shared by {self.model.config.num_attention_heads // self.model.config.num_key_value_heads} query heads")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def load_mmlu_dataset(self, json_path: str) -> List[Dict[str, Any]]:
        """
        Load questions from MMLU sample dataset JSON.
        
        Args:
            json_path: Path to the MMLU sample dataset JSON file
            
        Returns:
            List of question dictionaries
        """
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
            letter = chr(65 + i)  # A, B, C, D
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
        
        return prompt, incorrect_letter
    
    def run_inference_and_capture_keys(self, prompt: str, target_layers: List[int], 
                                       token_range: Tuple[int, int],
                                       max_new_tokens: int = 500) -> Dict[str, Any]:
        """
        Run inference while capturing key values at multiple layers for a token range.
        
        Args:
            prompt: Input prompt
            target_layers: List of layer indices to capture keys from (0 to num_layers - 1)
            token_range: Tuple of (start_position, end_position) for tokens to capture
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Dictionary containing response, tokens, and captured keys per layer
        """
        start_pos, end_pos = token_range
        
        print("\n🤖 Running inference with key capture...")
        print(f"   Target layers: {target_layers}")
        print(f"   Token range: {start_pos} to {end_pos} (inclusive)")
        print(f"   Will capture keys for {end_pos - start_pos + 1} tokens")
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        self.prompt_token_count = inputs['input_ids'].shape[1]
        
        print(f"   Prompt has {self.prompt_token_count} tokens")
        
        # Use model.generate with special output format to capture keys
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.1,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.0,
                return_dict_in_generate=True,
                output_hidden_states=True,
                use_cache=True
            )
        
        # Extract generated token IDs
        generated_ids = outputs.sequences[0]
        
        # Decode full sequence and response
        full_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
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
        
        print(f"   Generated {len(generated_ids) - self.prompt_token_count} new tokens")
        print(f"   Total sequence length: {len(generated_ids)} tokens")
        
        # Validate range
        if end_pos >= len(generated_ids):
            print(f"   ⚠️  Warning: End position {end_pos} exceeds total tokens {len(generated_ids)}")
            print(f"   Adjusting end position to {len(generated_ids) - 1}")
            end_pos = len(generated_ids) - 1
        
        # Extract keys from multiple layers using a forward pass
        print(f"\n📊 Extracting keys for layers {target_layers}, tokens {start_pos} to {end_pos}...")
        print(f"   Running forward pass to extract keys from attention layers...")
        
        # Extract keys for each layer
        keys_per_layer = {}
        for layer_idx in target_layers:
            print(f"   Extracting from layer {layer_idx}...")
            extracted_keys = self._extract_keys_from_forward_pass(
                generated_ids, layer_idx, start_pos, end_pos
            )
            keys_per_layer[layer_idx] = extracted_keys
        
        return {
            'prompt': prompt,
            'response': response_text,
            'full_text': full_text,
            'all_tokens': all_tokens,
            'prompt_token_count': self.prompt_token_count,
            'total_token_count': len(generated_ids),
            'token_range': (start_pos, end_pos),
            'target_layers': target_layers,
            'keys_per_layer': keys_per_layer,
            'generated_ids': generated_ids
        }
    
    def _extract_keys_from_forward_pass(self, token_ids: torch.Tensor, 
                                        target_layer: int,
                                        start_pos: int, end_pos: int) -> Dict[str, Any]:
        """
        Extract keys by running a forward pass and accessing the KV cache.
        
        Args:
            token_ids: Full sequence of token IDs
            target_layer: Layer to extract keys from
            start_pos: Start position (inclusive)
            end_pos: End position (inclusive)
            
        Returns:
            Dictionary with keys organized by token position and head
        """
        # Run forward pass with use_cache=True to get past_key_values
        with torch.no_grad():
            input_ids = token_ids.unsqueeze(0).to(self.device)
            outputs = self.model(
                input_ids, 
                output_attentions=False, 
                use_cache=True,
                return_dict=True
            )
        
        # Access past_key_values from the output
        # past_key_values is a tuple of length num_layers
        # Each element is a tuple of (keys, values) for that layer
        # keys shape: [batch, num_kv_heads, seq_len, head_dim]
        
        if not hasattr(outputs, 'past_key_values') or outputs.past_key_values is None:
            print(f"   ⚠️  Warning: Model output does not contain past_key_values")
            return {}
        
        past_key_values = outputs.past_key_values
        
        if target_layer >= len(past_key_values):
            print(f"   ⚠️  Warning: Layer {target_layer} out of range (only {len(past_key_values)} layers)")
            return {}
        
        # Get keys for the target layer
        layer_kv = past_key_values[target_layer]
        
        if not isinstance(layer_kv, tuple) or len(layer_kv) < 2:
            print(f"   ⚠️  Warning: Unexpected past_key_values format at layer {target_layer}")
            return {}
        
        # Extract keys (first element of the tuple)
        keys = layer_kv[0].detach().cpu()  # Shape: [batch, num_kv_heads, seq_len, head_dim]
        
        # Organize the captured keys
        batch_size, num_kv_heads, seq_len, head_dim = keys.shape
        
        organized_keys = {}
        for token_pos in range(seq_len):
            if start_pos <= token_pos <= end_pos:
                organized_keys[token_pos] = {}
                for head_idx in range(num_kv_heads):
                    key_vector = keys[0, head_idx, token_pos, :].numpy()
                    organized_keys[token_pos][f"kv_head_{head_idx}"] = key_vector.tolist()
        
        num_positions = len(organized_keys)
        expected = (end_pos - start_pos + 1)
        
        print(f"   ✅ Extracted keys for {num_positions} token positions")
        print(f"   Expected: {expected}, Got: {num_positions}")
        print(f"   Number of KV heads: {num_kv_heads}")
        print(f"   Head dimension: {head_dim}")
        print(f"   Note: Gemma-2 uses Grouped-Query Attention (GQA)")
        print(f"         {self.model.config.num_attention_heads} query heads share {num_kv_heads} KV heads")
        
        return organized_keys
    
    def save_results(self, result: Dict[str, Any], question_data: Dict[str, Any],
                    prompt_type: str, output_dir: str = "key_values") -> str:
        """
        Save key extraction results to a single JSON file containing all layers.
        
        Args:
            result: Inference result dictionary with keys per layer
            question_data: Original question data from MMLU
            prompt_type: Type of prompt used ('normal', 'correct_hint', 'incorrect_hint')
            output_dir: Directory to save results
            
        Returns:
            Path to saved file
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        start_pos, end_pos = result['token_range']
        target_layers = result['target_layers']
        keys_per_layer = result['keys_per_layer']
        
        # Create filename that reflects all layers
        if len(target_layers) == 1:
            layers_str = f"layer{target_layers[0]}"
        else:
            layers_str = f"layers{target_layers[0]}-{target_layers[-1]}" if len(target_layers) > 3 else f"layers{'_'.join(map(str, target_layers))}"
        
        filename = f"keys_{prompt_type}_{layers_str}_range_{start_pos}_{end_pos}_{timestamp}.json"
        output_file = os.path.join(output_dir, filename)
        
        # Prepare metadata
        metadata = {
            'model_name': self.model_name,
            'analysis_date': datetime.now().isoformat(),
            'device': self.device,
            'num_layers': self.model.config.num_hidden_layers,
            'target_layers': target_layers,
            'num_attention_heads': self.model.config.num_attention_heads,
            'num_key_value_heads': self.model.config.num_key_value_heads,
            'head_dim': self.model.config.hidden_size // self.model.config.num_attention_heads,
            'hidden_size': self.model.config.hidden_size,
            'prompt_type': prompt_type,
            'token_range': result['token_range']
        }
        
        # Prepare token info
        token_info = {
            'prompt_token_count': result['prompt_token_count'],
            'total_token_count': result['total_token_count'],
            'token_range': result['token_range'],
            'saved_token_count': end_pos - start_pos + 1,
            'all_tokens': result['all_tokens']
        }
        
        # Generate description
        if len(target_layers) == 1:
            layers_desc = f"layer {target_layers[0]}"
        else:
            layers_desc = f"layers {target_layers}"
        
        description = (
            f"Attention key values at {layers_desc} "
            f"for tokens in range [{start_pos}, {end_pos}] (inclusive). "
            f"Keys are organized by layer, then by token position and KV head. "
            f"Each key vector has dimension {metadata['head_dim']}. "
            f"The model uses Grouped-Query Attention (GQA): "
            f"{metadata['num_attention_heads']} query heads share "
            f"{metadata['num_key_value_heads']} key-value heads. "
            f"Each KV head is shared by {metadata['num_attention_heads'] // metadata['num_key_value_heads']} query heads."
        )
        
        # Prepare output data with all layers
        output_data = {
            'metadata': metadata,
            'question_info': {
                'question': question_data['question'],
                'choices': question_data['choices'],
                'correct_answer_idx': question_data['answer'],
                'correct_answer_letter': chr(65 + question_data['answer']),
                'subject': question_data['subject']
            },
            'prompt': result['prompt'],
            'response': result['response'],
            'full_text': result['full_text'],
            'token_info': token_info,
            'keys_per_layer': keys_per_layer,  # All layers in one structure
            'description': description
        }
        
        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results saved to: {filename}")
        print(f"   File size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")
        print(f"   Layers included: {target_layers}")
        
        return output_file


def main():
    """Main function with CLI interface."""
    import sys
    
    # Default parameters
    default_mmlu_path = "mmlu_sample_dataset.json"
    
    # Command Line Arguments
    if len(sys.argv) > 1:
        if sys.argv[1] in ["--help", "-h"]:
            print("\n" + "=" * 80)
            print("GEMMA-2-9B KEY VALUE EXTRACTION")
            print("=" * 80)
            print("\nUsage:")
            print("  python key_extraction.py <mmlu_path> <question_idx> <prompt_type> <layers> <num_tokens> [max_tokens]")
            print("\nArguments:")
            print("  mmlu_path       Path to MMLU sample dataset JSON file")
            print("  question_idx    Question index (0-based)")
            print("  prompt_type     'normal', 'correct_hint', or 'incorrect_hint'")
            print("  layers          Layer indices to extract keys from:")
            print("                  - Single layer: '20'")
            print("                  - Multiple layers (comma-separated): '1,10,20,30,40'")
            print("                  - Range notation: '10-15' (extracts layers 10,11,12,13,14,15)")
            print("  num_tokens      Number of tokens to capture after prompt ends")
            print("  max_tokens      (optional) Maximum number of tokens to generate [default: 500]")
            print("\nDescription:")
            print("  Extracts attention key values at specified layer(s) for N tokens after the prompt.")
            print("  Keys are captured for all KV heads at each specified layer.")
            print("  Saves one JSON file per layer for use in activation patching experiments.")
            print("  Note: Gemma-2 uses Grouped-Query Attention (GQA) with 8 KV heads shared by 16 query heads.")
            print("\nExamples:")
            print("  # Extract keys from layer 20 for first 5 tokens after prompt")
            print("  python key_extraction.py mmlu_sample_dataset.json 0 normal 20 5")
            print()
            print("  # Extract keys from layers 1,10,20,30,40 simultaneously")
            print("  python key_extraction.py mmlu_sample_dataset.json 0 normal 1,10,20,30,40 5")
            print()
            print("  # Extract keys from layers 10 through 15 (inclusive)")
            print("  python key_extraction.py mmlu_sample_dataset.json 0 normal 10-15 5")
            print()
            print("  # Extract with longer generation")
            print("  python key_extraction.py mmlu_sample_dataset.json 3 incorrect_hint 20,25,30 8 600")
            print("=" * 80)
            sys.exit(0)
        
        if len(sys.argv) < 6:
            print("❌ Error: Insufficient arguments")
            print("Run 'python key_extraction.py --help' for usage information")
            sys.exit(1)
        
        mmlu_path = sys.argv[1]
        question_idx = int(sys.argv[2])
        prompt_type = sys.argv[3]
        layers_arg = sys.argv[4]
        num_tokens_after_prompt = int(sys.argv[5])
        max_tokens = int(sys.argv[6]) if len(sys.argv) > 6 else 500
        
        # Parse layers argument
        target_layers = []
        if '-' in layers_arg and layers_arg.count('-') == 1:
            # Range notation: "10-15"
            try:
                start, end = layers_arg.split('-')
                start_layer = int(start)
                end_layer = int(end)
                target_layers = list(range(start_layer, end_layer + 1))
            except ValueError:
                print(f"❌ Error: Invalid range format '{layers_arg}'")
                print("   Range should be like '10-15'")
                sys.exit(1)
        elif ',' in layers_arg:
            # Comma-separated: "1,10,20,30"
            try:
                target_layers = [int(x.strip()) for x in layers_arg.split(',')]
            except ValueError:
                print(f"❌ Error: Invalid comma-separated layers '{layers_arg}'")
                sys.exit(1)
        else:
            # Single layer: "20"
            try:
                target_layers = [int(layers_arg)]
            except ValueError:
                print(f"❌ Error: Invalid layer number '{layers_arg}'")
                sys.exit(1)
    else:
        print("❌ Error: No arguments provided")
        print("Run 'python key_extraction.py --help' for usage information")
        sys.exit(1)
    
    # Run the extraction
    try:
        print("=" * 80)
        print("GEMMA-2-9B KEY VALUE EXTRACTION")
        print("=" * 80)
        print(f"Configuration:")
        print(f"  Question index: {question_idx}")
        print(f"  Prompt type: {prompt_type}")
        print(f"  Target layers: {target_layers}")
        print(f"  Number of layers: {len(target_layers)}")
        print(f"  Tokens to capture: {num_tokens_after_prompt}")
        print(f"  Max tokens to generate: {max_tokens}")
        
        # Initialize extractor
        extractor = GemmaKeyValueExtractor(device="auto")
        
        # Load tokenizer for prompt length calculation
        print(f"\nLoading tokenizer...")
        from transformers import AutoTokenizer
        extractor.tokenizer = AutoTokenizer.from_pretrained(extractor.model_name)
        print("✅ Tokenizer loaded")
        
        # Load MMLU dataset
        questions = extractor.load_mmlu_dataset(mmlu_path)
        
        # Validate question index
        if question_idx >= len(questions):
            print(f"❌ Error: Question index {question_idx} out of range (0-{len(questions)-1})")
            sys.exit(1)
        
        question_data = questions[question_idx]
        
        # Format prompt based on type
        print(f"\n📋 Formatting prompt with type: {prompt_type}")
        
        if prompt_type == "normal":
            prompt = extractor.format_question_prompt(question_data)
        elif prompt_type == "correct_hint":
            prompt, _ = extractor.format_question_prompt_correct_hint(question_data)
        elif prompt_type == "incorrect_hint":
            prompt, _ = extractor.format_question_prompt_incorrect_hint(question_data)
        else:
            print(f"❌ Error: Invalid prompt type '{prompt_type}'")
            print("   Valid types: 'normal', 'correct_hint', 'incorrect_hint'")
            sys.exit(1)
        
        # Tokenize prompt to determine length
        print("\n🔤 Tokenizing prompt to determine start position...")
        inputs = extractor.tokenizer(prompt, return_tensors="pt")
        prompt_length = inputs['input_ids'].shape[1]
        print(f"   Prompt length: {prompt_length} tokens")
        print(f"   Last prompt token position: {prompt_length - 1}")
        
        # Calculate token range
        start_pos = prompt_length - 1
        end_pos = start_pos + num_tokens_after_prompt - 1
        
        print(f"\n📍 Calculated token range: [{start_pos}, {end_pos}]")
        print(f"   These keys will be used to predict tokens at positions {start_pos+1} to {end_pos+1}")
        
        # Validate layer indices
        extractor.load_model()
        
        max_layer = extractor.model.config.num_hidden_layers - 1
        invalid_layers = [l for l in target_layers if l < 0 or l > max_layer]
        if invalid_layers:
            print(f"❌ Error: Layer indices {invalid_layers} out of range (0-{max_layer})")
            sys.exit(1)
        
        # Run inference and capture keys from all layers
        result = extractor.run_inference_and_capture_keys(
            prompt,
            target_layers,
            token_range=(start_pos, end_pos),
            max_new_tokens=max_tokens
        )
        
        print("\n" + "=" * 80)
        print("MODEL RESPONSE:")
        print("=" * 80)
        print(result['response'])
        print("\n" + "=" * 80)
        
        # Save results (all layers in one file)
        print("\n" + "=" * 80)
        print("SAVING RESULTS")
        print("=" * 80)
        
        output_file = extractor.save_results(result, question_data, prompt_type)
        
        print("\n" + "=" * 80)
        print("✅ EXTRACTION COMPLETE!")
        print("=" * 80)
        print(f"Extracted keys from {len(target_layers)} layer(s): {target_layers}")
        print(f"Saved to: {os.path.basename(output_file)}")
        print(f"\nSummary:")
        print(f"  - Question: {question_data['question'][:60]}...")
        print(f"  - Prompt type: {prompt_type}")
        print(f"  - Target layers: {target_layers}")
        print(f"  - Prompt length: {prompt_length} tokens")
        print(f"  - Total tokens generated: {result['total_token_count']}")
        print(f"  - Token range captured: [{start_pos}, {end_pos}]")
        print(f"  - Tokens saved per layer: {num_tokens_after_prompt}")
        print(f"  - Architecture: Grouped-Query Attention (GQA)")
        print(f"    • Query heads: {extractor.model.config.num_attention_heads}")
        print(f"    • KV heads captured: {extractor.model.config.num_key_value_heads}")
        print(f"    • Head dimension: {extractor.model.config.hidden_size // extractor.model.config.num_attention_heads}")
        print(f"    • Sharing ratio: {extractor.model.config.num_attention_heads // extractor.model.config.num_key_value_heads} query heads per KV head")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Extraction interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

