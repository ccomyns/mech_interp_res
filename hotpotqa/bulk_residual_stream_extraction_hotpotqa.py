#!/usr/bin/env python3
"""
Bulk Residual Stream Extraction for Gemma-3-4B on HotPotQA

This script processes HotPotQA questions in bulk and extracts residual stream
hidden states for each. Results are saved to a single JSON file containing:
- Residual stream values for each question
- Raw model responses for each question

Usage (Index mode):
    python bulk_residual_stream_extraction_hotpotqa.py <hotpotqa_path> <num_tokens_after_prompt> --start-idx START --end-idx END [--context-mode MODE]

Usage (Per-level mode):
    python bulk_residual_stream_extraction_hotpotqa.py <hotpotqa_path> <num_tokens_after_prompt> --per-level EASY MEDIUM HARD [--context-mode MODE]

Arguments:
    hotpotqa_path          Path to HotPotQA dataset JSON file
    num_tokens_after_prompt Number of tokens to capture after prompt ends
    --start-idx            Starting question index (inclusive) - for index mode
    --end-idx              Ending question index (exclusive) - for index mode
    --per-level            Number of questions per difficulty level (easy medium hard) - for per-level mode
    --context-mode         How to include context: 'full' (all paragraphs), 'supporting' (only supporting), 
                           'shuffled' (all paragraphs shuffled). Default: 'full'

Examples:
    # Index mode: Process questions 0-9, capture 5 tokens after each prompt
    python bulk_residual_stream_extraction_hotpotqa.py hotpotqa_train.json 5 --start-idx 0 --end-idx 10
    
    # Per-level mode: Process 100 easy, 100 medium, 100 hard questions
    python bulk_residual_stream_extraction_hotpotqa.py hotpotqa_train.json 5 --per-level 100 100 100
    
    # Per-level mode with supporting context only
    python bulk_residual_stream_extraction_hotpotqa.py hotpotqa_train.json 5 --per-level 50 50 50 --context-mode supporting
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
    """Analyzer for extracting residual stream hidden states from Gemma-3-4B."""
    
    def __init__(self, model_name: str = "google/gemma-3-4b-it", device: str = "cpu"):
        """
        Initialize the Gemma-3-4B residual stream analyzer.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run inference on ('auto', 'cuda', 'cpu')
        """
        self.model_name = model_name
        self.device = self._setup_device(device)
        self.tokenizer = None
        self.model = None
        self.prompt_token_count = 0
        self._num_layers = None
        self._hidden_size = None
    
    def _get_config_attr(self, config, attr_names: list, default=None):
        """
        Get a config attribute, trying multiple possible names.
        Gemma 3 may use text_config for text-specific attributes.
        """
        # First try the main config
        for name in attr_names:
            if hasattr(config, name):
                return getattr(config, name)
        
        # For Gemma 3 multimodal, try text_config
        if hasattr(config, 'text_config'):
            for name in attr_names:
                if hasattr(config.text_config, name):
                    return getattr(config.text_config, name)
        
        return default
        
    def _setup_device(self, device: str) -> str:
        """Setup the appropriate device for inference."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return device
    
    def load_model(self):
        """Load the Gemma-3-4B model and tokenizer with hidden states output enabled."""
        print(f"Loading {self.model_name} on {self.device}...")
        print("This may take a few minutes...")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Load model with appropriate settings
            if self.device == "cuda":
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.bfloat16,  # bfloat16 avoids overflow (same range as fp32)
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
            # Get num_layers and hidden_size (Gemma 3 uses different attribute names and may nest under text_config)
            self._num_layers = self._get_config_attr(
                self.model.config, 
                ['num_hidden_layers', 'num_layers', 'n_layer', 'n_layers']
            )
            self._hidden_size = self._get_config_attr(
                self.model.config,
                ['hidden_size', 'hidden_dim', 'd_model', 'model_dim']
            )
            print(f"Model has {self._num_layers} transformer layers + 1 embedding layer")
            print(f"Hidden size: {self._hidden_size}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def load_hotpotqa_dataset(self, json_path: str) -> List[Dict[str, Any]]:
        """Load questions from HotPotQA dataset JSON."""
        print(f"\n📂 Loading HotPotQA dataset from: {json_path}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # HotPotQA is a list of questions directly
        questions = data if isinstance(data, list) else data.get('questions', data)
        print(f"✅ Loaded {len(questions)} questions")
        
        return questions
    
    def format_question_prompt_full_context(self, question_data: Dict[str, Any]) -> str:
        """Format a HotPotQA question with full context (all paragraphs)."""
        question = question_data["question"]
        paragraphs = question_data["paragraphs"]
        
        prompt = "You are answering a reading comprehension question. Read the following passages carefully and answer the question based on the information provided.\n\n"
        prompt += "=== PASSAGES ===\n\n"
        
        for para in paragraphs:
            title = para.get("title", "")
            text = para.get("text", "")
            prompt += f"[{title}]\n{text}\n\n"
        
        prompt += "=== QUESTION ===\n"
        prompt += f"{question}\n\n"
        prompt += "Instructions: Answer the question based on the passages above. "
        prompt += "Think through your reasoning step by step, then provide your final answer.\n\n"
        prompt += "Your response: "
        
        return prompt
    
    def format_question_prompt_supporting_only(self, question_data: Dict[str, Any]) -> str:
        """Format a HotPotQA question with only supporting paragraphs."""
        question = question_data["question"]
        paragraphs = question_data["paragraphs"]
        
        # Filter to only supporting paragraphs
        supporting_paragraphs = [p for p in paragraphs if p.get("is_supporting", False)]
        
        prompt = "You are answering a reading comprehension question. Read the following passages carefully and answer the question based on the information provided.\n\n"
        prompt += "=== PASSAGES ===\n\n"
        
        for para in supporting_paragraphs:
            title = para.get("title", "")
            text = para.get("text", "")
            prompt += f"[{title}]\n{text}\n\n"
        
        prompt += "=== QUESTION ===\n"
        prompt += f"{question}\n\n"
        prompt += "Instructions: Answer the question based on the passages above. "
        prompt += "Think through your reasoning step by step, then provide your final answer.\n\n"
        prompt += "Your response: "
        
        return prompt
    
    def format_question_prompt_shuffled(self, question_data: Dict[str, Any]) -> str:
        """Format a HotPotQA question with shuffled paragraphs."""
        question = question_data["question"]
        paragraphs = question_data["paragraphs"].copy()
        
        # Shuffle the paragraphs
        random.shuffle(paragraphs)
        
        prompt = "You are answering a reading comprehension question. Read the following passages carefully and answer the question based on the information provided.\n\n"
        prompt += "=== PASSAGES ===\n\n"
        
        for para in paragraphs:
            title = para.get("title", "")
            text = para.get("text", "")
            prompt += f"[{title}]\n{text}\n\n"
        
        prompt += "=== QUESTION ===\n"
        prompt += f"{question}\n\n"
        prompt += "Instructions: Answer the question based on the passages above. "
        prompt += "Think through your reasoning step by step, then provide your final answer.\n\n"
        prompt += "Your response: "
        
        return prompt
    
    def run_inference_with_hidden_states_for_range(self, prompt: str, token_range: Tuple[int, int],
                                                   max_new_tokens: int = 1500) -> Dict[str, Any]:
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
        """Extract hidden states from the generation output for a specific token range.
        
        Only captures layer 18:
        - Layer 18: Midpoint layer (middle of the 34-layer Gemma-3-4B model)
        """
        if not hasattr(outputs, 'hidden_states') or outputs.hidden_states is None:
            raise ValueError("No hidden states available in output")
        
        # Only capture layer 18 (midpoint)
        layers_to_capture = [18]
        
        # Initialize storage for hidden states (only for specified layers)
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


def check_answer_in_response(response: str, expected_answer: str) -> Dict[str, Any]:
    """
    Check if the expected answer appears in the response.
    
    Args:
        response: The model's raw response text
        expected_answer: The expected answer string
        
    Returns:
        Dictionary with answer_found flag and related info
    """
    # Normalize strings for comparison
    response_lower = response.lower()
    answer_lower = expected_answer.lower()
    
    # Check for exact match
    exact_match = answer_lower in response_lower
    
    # Check for word-boundary match (the answer appears as whole words)
    import re
    word_boundary_match = bool(re.search(r'\b' + re.escape(answer_lower) + r'\b', response_lower))
    
    return {
        'answer_found_exact': exact_match,
        'answer_found_word_boundary': word_boundary_match,
        'expected_answer': expected_answer,
        'needs_review': not word_boundary_match
    }


def select_questions_by_level(questions: List[Dict[str, Any]], 
                               num_easy: int, num_medium: int, num_hard: int) -> List[Tuple[int, Dict[str, Any]]]:
    """
    Select a specified number of questions from each difficulty level.
    
    Args:
        questions: Full list of questions from the dataset
        num_easy: Number of easy questions to select
        num_medium: Number of medium questions to select
        num_hard: Number of hard questions to select
        
    Returns:
        List of (original_index, question_data) tuples
    """
    # Group questions by level
    easy_questions = [(i, q) for i, q in enumerate(questions) if q.get('level', '').lower() == 'easy']
    medium_questions = [(i, q) for i, q in enumerate(questions) if q.get('level', '').lower() == 'medium']
    hard_questions = [(i, q) for i, q in enumerate(questions) if q.get('level', '').lower() == 'hard']
    
    print(f"\n📊 Dataset composition:")
    print(f"   Easy questions available: {len(easy_questions)}")
    print(f"   Medium questions available: {len(medium_questions)}")
    print(f"   Hard questions available: {len(hard_questions)}")
    
    # Validate requested counts
    if num_easy > len(easy_questions):
        print(f"⚠️  Warning: Requested {num_easy} easy questions but only {len(easy_questions)} available")
        num_easy = len(easy_questions)
    if num_medium > len(medium_questions):
        print(f"⚠️  Warning: Requested {num_medium} medium questions but only {len(medium_questions)} available")
        num_medium = len(medium_questions)
    if num_hard > len(hard_questions):
        print(f"⚠️  Warning: Requested {num_hard} hard questions but only {len(hard_questions)} available")
        num_hard = len(hard_questions)
    
    # Select questions (take first N from each level for reproducibility)
    selected = []
    selected.extend(easy_questions[:num_easy])
    selected.extend(medium_questions[:num_medium])
    selected.extend(hard_questions[:num_hard])
    
    print(f"\n✅ Selected questions:")
    print(f"   Easy: {num_easy}")
    print(f"   Medium: {num_medium}")
    print(f"   Hard: {num_hard}")
    print(f"   Total: {len(selected)}")
    
    return selected


def run_bulk_extraction(hotpotqa_path: str, start_idx: int, end_idx: int,
                        num_tokens_after_prompt: int, context_mode: str = "full",
                        output_dir: str = "hotpotqa") -> str:
    """
    Run bulk residual stream extraction for a range of HotPotQA questions (index mode).
    
    Args:
        hotpotqa_path: Path to HotPotQA dataset JSON
        start_idx: Starting question index (inclusive)
        end_idx: Ending question index (exclusive)
        num_tokens_after_prompt: Number of tokens to capture after prompt ends
        context_mode: How to include context ('full', 'supporting', 'shuffled')
        output_dir: Directory to save results
        
    Returns:
        Path to saved output file
    """
    print("=" * 80)
    print("BULK RESIDUAL STREAM EXTRACTION - HotPotQA (Index Mode)")
    print("=" * 80)
    print(f"Question range: [{start_idx}, {end_idx})")
    print(f"Tokens after prompt: {num_tokens_after_prompt}")
    print(f"Context mode: {context_mode}")
    print("=" * 80)
    
    # Initialize analyzer
    analyzer = GemmaResidualStreamAnalyzer(device="auto")
    analyzer.load_model()
    
    # Load HotPotQA dataset
    questions = analyzer.load_hotpotqa_dataset(hotpotqa_path)
    
    # Validate indices
    if start_idx < 0:
        raise ValueError(f"start_idx must be >= 0, got {start_idx}")
    if end_idx > len(questions):
        print(f"⚠️  Warning: end_idx {end_idx} exceeds dataset size {len(questions)}")
        end_idx = len(questions)
    if start_idx >= end_idx:
        raise ValueError(f"start_idx ({start_idx}) must be < end_idx ({end_idx})")
    
    # Create list of (index, question) tuples
    selected_questions = [(i, questions[i]) for i in range(start_idx, end_idx)]
    num_questions = len(selected_questions)
    print(f"\n📊 Processing {num_questions} questions...")
    
    # Run the common extraction logic
    results = _process_questions(analyzer, selected_questions, num_tokens_after_prompt, context_mode)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"bulk_residual_stream_hotpotqa_{context_mode}_q{start_idx}-{end_idx-1}_tokens{num_tokens_after_prompt}_{timestamp}.json"
    output_file = os.path.join(output_dir, filename)
    
    # Prepare output data
    output_data = {
        'metadata': {
            'model_name': analyzer.model_name,
            'analysis_date': datetime.now().isoformat(),
            'device': analyzer.device,
            'num_layers': analyzer._num_layers,
            'hidden_size': analyzer._hidden_size,
            'context_mode': context_mode,
            'selection_mode': 'index_range',
            'question_range': {
                'start_idx': start_idx,
                'end_idx': end_idx,
                'num_questions': num_questions
            },
            'num_tokens_after_prompt': num_tokens_after_prompt,
            'hotpotqa_path': hotpotqa_path
        },
        'description': (
            f"Bulk extraction of residual stream hidden states for HotPotQA questions {start_idx} to {end_idx-1}. "
            f"Context mode: {context_mode}. "
            f"For each question, hidden states are captured at layer_18 (midpoint layer) "
            f"for {num_tokens_after_prompt} tokens after the prompt ends. "
            f"Each result includes the raw model response and the residual stream values."
        ),
        'layers_captured': [18],
        'results': results
    }
    
    # Save and print summary
    return _save_and_summarize(output_file, output_data, analyzer, num_questions, results, context_mode, num_tokens_after_prompt)


def run_bulk_extraction_by_level(hotpotqa_path: str, num_easy: int, num_medium: int, num_hard: int,
                                  num_tokens_after_prompt: int, context_mode: str = "full",
                                  output_dir: str = "hotpotqa") -> str:
    """
    Run bulk residual stream extraction for a specified number of questions per difficulty level.
    
    Args:
        hotpotqa_path: Path to HotPotQA dataset JSON
        num_easy: Number of easy questions to process
        num_medium: Number of medium questions to process
        num_hard: Number of hard questions to process
        num_tokens_after_prompt: Number of tokens to capture after prompt ends
        context_mode: How to include context ('full', 'supporting', 'shuffled')
        output_dir: Directory to save results
        
    Returns:
        Path to saved output file
    """
    print("=" * 80)
    print("BULK RESIDUAL STREAM EXTRACTION - HotPotQA (Per-Level Mode)")
    print("=" * 80)
    print(f"Requested: {num_easy} easy, {num_medium} medium, {num_hard} hard")
    print(f"Tokens after prompt: {num_tokens_after_prompt}")
    print(f"Context mode: {context_mode}")
    print("=" * 80)
    
    # Initialize analyzer
    analyzer = GemmaResidualStreamAnalyzer(device="auto")
    analyzer.load_model()
    
    # Load HotPotQA dataset
    questions = analyzer.load_hotpotqa_dataset(hotpotqa_path)
    
    # Select questions by level
    selected_questions = select_questions_by_level(questions, num_easy, num_medium, num_hard)
    num_questions = len(selected_questions)
    
    if num_questions == 0:
        raise ValueError("No questions selected. Check that the dataset contains questions with 'level' field.")
    
    # Run the common extraction logic
    results = _process_questions(analyzer, selected_questions, num_tokens_after_prompt, context_mode)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Count actual questions per level in results
    actual_easy = sum(1 for r in results if r.get('question_info', {}).get('level', '').lower() == 'easy' and 'error' not in r)
    actual_medium = sum(1 for r in results if r.get('question_info', {}).get('level', '').lower() == 'medium' and 'error' not in r)
    actual_hard = sum(1 for r in results if r.get('question_info', {}).get('level', '').lower() == 'hard' and 'error' not in r)
    
    # Generate output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"bulk_residual_stream_hotpotqa_{context_mode}_e{actual_easy}m{actual_medium}h{actual_hard}_tokens{num_tokens_after_prompt}_{timestamp}.json"
    output_file = os.path.join(output_dir, filename)
    
    # Prepare output data
    output_data = {
        'metadata': {
            'model_name': analyzer.model_name,
            'analysis_date': datetime.now().isoformat(),
            'device': analyzer.device,
            'num_layers': analyzer._num_layers,
            'hidden_size': analyzer._hidden_size,
            'context_mode': context_mode,
            'selection_mode': 'per_level',
            'questions_per_level': {
                'requested': {'easy': num_easy, 'medium': num_medium, 'hard': num_hard},
                'actual': {'easy': actual_easy, 'medium': actual_medium, 'hard': actual_hard},
                'total': num_questions
            },
            'num_tokens_after_prompt': num_tokens_after_prompt,
            'hotpotqa_path': hotpotqa_path
        },
        'description': (
            f"Bulk extraction of residual stream hidden states for HotPotQA questions by difficulty level. "
            f"Requested: {num_easy} easy, {num_medium} medium, {num_hard} hard. "
            f"Context mode: {context_mode}. "
            f"For each question, hidden states are captured at layer_18 (midpoint layer) "
            f"for {num_tokens_after_prompt} tokens after the prompt ends. "
            f"Each result includes the raw model response and the residual stream values."
        ),
        'layers_captured': [18],
        'results': results
    }
    
    # Save and print summary
    return _save_and_summarize(output_file, output_data, analyzer, num_questions, results, context_mode, num_tokens_after_prompt,
                               level_counts={'easy': actual_easy, 'medium': actual_medium, 'hard': actual_hard})


def _process_questions(analyzer: GemmaResidualStreamAnalyzer, 
                       selected_questions: List[Tuple[int, Dict[str, Any]]],
                       num_tokens_after_prompt: int, 
                       context_mode: str) -> List[Dict[str, Any]]:
    """
    Common logic for processing a list of selected questions.
    
    Args:
        analyzer: Initialized GemmaResidualStreamAnalyzer
        selected_questions: List of (original_index, question_data) tuples
        num_tokens_after_prompt: Number of tokens to capture after prompt ends
        context_mode: How to include context
        
    Returns:
        List of result dictionaries
    """
    results = []
    num_questions = len(selected_questions)
    
    for i, (q_idx, question_data) in enumerate(selected_questions):
        print(f"\n{'='*80}")
        print(f"QUESTION {q_idx} ({i + 1}/{num_questions})")
        print("=" * 80)
        
        print(f"Type: {question_data.get('type', 'unknown')}")
        print(f"Level: {question_data.get('level', 'unknown')}")
        print(f"Question: {question_data['question'][:100]}...")
        print(f"Expected answer: {question_data['answer']}")
        
        # Format prompt based on context mode
        if context_mode == "full":
            prompt = analyzer.format_question_prompt_full_context(question_data)
        elif context_mode == "supporting":
            prompt = analyzer.format_question_prompt_supporting_only(question_data)
        elif context_mode == "shuffled":
            prompt = analyzer.format_question_prompt_shuffled(question_data)
        else:
            raise ValueError(f"Invalid context_mode: {context_mode}")
        
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
                max_new_tokens=1500
            )
            
            print(f"✅ Generated {result['total_token_count'] - result['prompt_token_count']} tokens")
            print(f"Response preview: {result['response'][:150]}...")
            
            # Check if expected answer appears in response
            answer_check = check_answer_in_response(result['response'], question_data['answer'])
            
            # Store result for this question
            question_result = {
                'question_idx': q_idx,
                'question_info': {
                    'id': question_data.get('id', f'q_{q_idx}'),
                    'question': question_data['question'],
                    'answer': question_data['answer'],
                    'type': question_data.get('type', 'unknown'),
                    'level': question_data.get('level', 'unknown'),
                    'num_supporting': question_data.get('num_supporting', 0),
                    'num_distractors': question_data.get('num_distractors', 0)
                },
                'context_mode': context_mode,
                'prompt': result['prompt'],
                'raw_response': result['response'],
                'full_text': result['full_text'],
                'answer_found_exact': answer_check['answer_found_exact'],
                'answer_found_word_boundary': answer_check['answer_found_word_boundary'],
                'needs_review': answer_check['needs_review'],
            }
            
            # Check if token budget was exceeded (generated tokens == max_new_tokens)
            generated_tokens = result['total_token_count'] - result['prompt_token_count']
            exceeded_budget = generated_tokens >= 1500
            question_result['exceeded_token_budget'] = exceeded_budget
            
            question_result['prompt_token_count'] = result['prompt_token_count']
            question_result['total_token_count'] = result['total_token_count']
            question_result['token_range'] = result['token_range']
            question_result['all_tokens'] = result['all_tokens']
            question_result['residual_stream_hidden_states'] = result['hidden_states']
            
            # Print answer check summary
            print(f"   Answer found (word boundary): {answer_check['answer_found_word_boundary']} | needs_review: {answer_check['needs_review']}")
            if exceeded_budget:
                print(f"   ⚠️  Token budget exceeded ({generated_tokens} tokens generated)")
            
            results.append(question_result)
            
        except Exception as e:
            print(f"❌ Error processing question {q_idx}: {e}")
            results.append({
                'question_idx': q_idx,
                'error': str(e)
            })
    
    return results


def _save_and_summarize(output_file: str, output_data: Dict[str, Any], 
                        analyzer: GemmaResidualStreamAnalyzer, num_questions: int,
                        results: List[Dict[str, Any]], context_mode: str, 
                        num_tokens_after_prompt: int, level_counts: Optional[Dict[str, int]] = None) -> str:
    """
    Save results to file and print summary.
    
    Args:
        output_file: Path to output file
        output_data: Data to save
        analyzer: The analyzer instance
        num_questions: Total number of questions processed
        results: List of results
        context_mode: Context mode used
        num_tokens_after_prompt: Number of tokens captured
        level_counts: Optional dict with counts per level (for per-level mode)
        
    Returns:
        Path to saved output file
    """
    # Save to file
    print(f"\n💾 Saving results to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    file_size = os.path.getsize(output_file) / (1024 * 1024)
    print(f"   File size: {file_size:.2f} MB")
    
    # Summary
    successful = sum(1 for r in results if 'error' not in r)
    failed = num_questions - successful
    answers_found = sum(1 for r in results if r.get('answer_found_word_boundary', False))
    
    print("\n" + "=" * 80)
    print("✅ BULK EXTRACTION COMPLETE!")
    print("=" * 80)
    print(f"Results saved to: {output_file}")
    print(f"\nSummary:")
    print(f"  - Questions processed: {num_questions}")
    print(f"  - Successful: {successful}")
    print(f"  - Failed: {failed}")
    if level_counts:
        print(f"  - By level: {level_counts['easy']} easy, {level_counts['medium']} medium, {level_counts['hard']} hard")
    print(f"  - Answers found in response: {answers_found}/{successful}")
    print(f"  - Context mode: {context_mode}")
    print(f"  - Tokens captured per question: {num_tokens_after_prompt}")
    print(f"  - Layers captured: 18 (midpoint)")
    print(f"  - Hidden size: {analyzer._hidden_size}")
    print("=" * 80)
    
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Bulk residual stream extraction for Gemma-3-4B on HotPotQA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Index mode: Process questions 0-9 with full context, capture 5 tokens
  python bulk_residual_stream_extraction_hotpotqa.py hotpotqa_train.json 5 --start-idx 0 --end-idx 10

  # Per-level mode: Process 100 easy, 100 medium, 100 hard questions
  python bulk_residual_stream_extraction_hotpotqa.py hotpotqa_train.json 5 --per-level 100 100 100

  # Per-level mode with supporting context only
  python bulk_residual_stream_extraction_hotpotqa.py hotpotqa_train.json 5 --per-level 50 50 50 --context-mode supporting

  # Index mode with shuffled paragraphs
  python bulk_residual_stream_extraction_hotpotqa.py hotpotqa_train.json 5 --start-idx 0 --end-idx 10 --context-mode shuffled
        """
    )
    
    parser.add_argument('hotpotqa_path', type=str,
                        help='Path to HotPotQA dataset JSON file')
    parser.add_argument('num_tokens_after_prompt', type=int,
                        help='Number of tokens to capture after prompt ends')
    
    # Index mode arguments
    parser.add_argument('--start-idx', type=int, default=None,
                        help='Starting question index (inclusive) - for index mode')
    parser.add_argument('--end-idx', type=int, default=None,
                        help='Ending question index (exclusive) - for index mode')
    
    # Per-level mode argument
    parser.add_argument('--per-level', type=int, nargs=3, metavar=('EASY', 'MEDIUM', 'HARD'),
                        help='Number of questions per difficulty level (easy medium hard) - for per-level mode')
    
    # Common arguments
    parser.add_argument('--context-mode', type=str, default='full',
                        choices=['full', 'supporting', 'shuffled'],
                        help="Context mode: 'full' (all paragraphs), 'supporting' (only supporting), 'shuffled' (all paragraphs shuffled). Default: 'full'")
    parser.add_argument('--output-dir', type=str, default='hotpotqa',
                        help='Output directory for results (default: hotpotqa)')
    
    args = parser.parse_args()
    
    # Validate that exactly one mode is specified
    index_mode = args.start_idx is not None or args.end_idx is not None
    level_mode = args.per_level is not None
    
    if index_mode and level_mode:
        parser.error("Cannot use both index mode (--start-idx/--end-idx) and per-level mode (--per-level) at the same time")
    
    if not index_mode and not level_mode:
        parser.error("Must specify either index mode (--start-idx and --end-idx) or per-level mode (--per-level)")
    
    if index_mode:
        if args.start_idx is None or args.end_idx is None:
            parser.error("Index mode requires both --start-idx and --end-idx")
    
    try:
        if level_mode:
            num_easy, num_medium, num_hard = args.per_level
            run_bulk_extraction_by_level(
                hotpotqa_path=args.hotpotqa_path,
                num_easy=num_easy,
                num_medium=num_medium,
                num_hard=num_hard,
                num_tokens_after_prompt=args.num_tokens_after_prompt,
                context_mode=args.context_mode,
                output_dir=args.output_dir
            )
        else:
            run_bulk_extraction(
                hotpotqa_path=args.hotpotqa_path,
                start_idx=args.start_idx,
                end_idx=args.end_idx,
                num_tokens_after_prompt=args.num_tokens_after_prompt,
                context_mode=args.context_mode,
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

