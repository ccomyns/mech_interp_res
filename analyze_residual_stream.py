#!/usr/bin/env python3
"""
Residual Stream Analysis Tool for Gemma-3-4B

This script analyzes the hidden states saved by bulk_residual_stream_extraction.py.
It performs:
1. Loading saved residual stream data
2. Normalization of hidden states (final layer norm)
3. Unembedding via model.lm_head to get logits
4. Computing probability distributions over vocabulary
5. Showing top-k predicted tokens for any position/layer
"""

import json
import os
from typing import Dict, List, Any, Tuple, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from pathlib import Path


class ResidualStreamAnalyzer:
    def __init__(self, model_name: str = "google/gemma-3-4b-it", device: str = "auto"):
        """
        Initialize the residual stream analyzer for Gemma-3-4B.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run analysis on ('auto', 'cuda', 'cpu')
        """
        self.model_name = model_name
        self.device = self._setup_device(device)
        self.tokenizer = None
        self.model = None
        self.loaded_data = None
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
        """Setup the appropriate device."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return device
    
    def load_model(self):
        """
        Load the Gemma-3-4B model and tokenizer.
        We need access to model.model.norm (final layer norm) and model.lm_head (unembedding).
        """
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
            
            self.model.eval()  # Set to evaluation mode
            
            print(f"✅ Model loaded successfully on {self.device}")
            # Get config attributes (Gemma 3 uses different names and may nest under text_config)
            self._num_layers = self._get_config_attr(
                self.model.config,
                ['num_hidden_layers', 'num_layers', 'n_layer', 'n_layers']
            )
            self._hidden_size = self._get_config_attr(
                self.model.config,
                ['hidden_size', 'hidden_dim', 'd_model', 'model_dim']
            )
            vocab_size = self._get_config_attr(
                self.model.config,
                ['vocab_size']
            )
            print(f"   Vocabulary size: {vocab_size}")
            print(f"   Hidden size: {self._hidden_size}")
            print(f"   Number of layers: {self._num_layers}")
            
            # Verify we have access to the necessary components
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'norm'):
                print(f"   ✅ Found final layer norm: model.model.norm")
            else:
                print(f"   ⚠️  Warning: Could not find model.model.norm")
                
            if hasattr(self.model, 'lm_head'):
                print(f"   ✅ Found language model head: model.lm_head")
            else:
                print(f"   ⚠️  Warning: Could not find model.lm_head")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def load_residual_stream_data(self, json_path: str) -> Dict[str, Any]:
        """
        Load residual stream data from JSON file.
        
        Args:
            json_path: Path to the residual stream JSON file
            
        Returns:
            Dictionary with all the saved data
        """
        print(f"\n📂 Loading residual stream data from: {json_path}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.loaded_data = data
        
        print(f"✅ Data loaded successfully")
        print(f"   Model: {data['metadata']['model_name']}")
        print(f"   Prompt type: {data['metadata']['prompt_type']}")
        print(f"   Number of layers: {data['metadata']['num_layers']}")
        print(f"   Hidden size: {data['metadata']['hidden_size']}")
        print(f"   Total tokens: {data['token_info']['total_token_count']}")
        
        # Handle both cutoff mode and range mode
        mode = data['metadata'].get('mode', 'cutoff')  # Default to cutoff for backwards compatibility
        if mode == 'range':
            start_pos, end_pos = data['metadata']['token_range']
            print(f"   Mode: Token range")
            print(f"   Token range: [{start_pos}, {end_pos}]")
            print(f"   Tokens saved: {data['token_info']['saved_token_count']}")
        else:
            print(f"   Mode: Cutoff")
            print(f"   CoT tokens saved: {data['token_info']['cot_token_count']}")
            print(f"   Cutoff position: {data['metadata']['cutoff_position']}")
        
        return data
    
    def get_available_token_positions(self, data: Optional[Dict[str, Any]] = None) -> List[int]:
        """
        Get list of token positions that have saved hidden states.
        
        Args:
            data: Loaded data dictionary (uses self.loaded_data if None)
            
        Returns:
            Sorted list of token positions
        """
        if data is None:
            data = self.loaded_data
        
        if data is None:
            raise ValueError("No data loaded. Call load_residual_stream_data() first.")
        
        # Get positions from any layer (they should all be the same)
        layer_key = "layer_0"
        hidden_states = data['residual_stream_hidden_states'][layer_key]
        positions = sorted([int(pos) for pos in hidden_states.keys()])
        
        return positions
    
    def normalize_and_unembed(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Apply final layer normalization and unembedding to get logits.
        
        Args:
            hidden_state: Hidden state tensor of shape [hidden_size] or [1, hidden_size]
            
        Returns:
            Logits tensor of shape [vocab_size]
        """
        # Ensure we have the right shape
        if hidden_state.dim() == 1:
            hidden_state = hidden_state.unsqueeze(0)  # [1, hidden_size]
        
        # Apply final layer normalization
        # For Gemma-3, this is model.model.norm
        with torch.no_grad():
            normalized = self.model.model.norm(hidden_state)
            
            # Apply language model head (unembedding)
            logits = self.model.lm_head(normalized)
        
        # Return as 1D tensor
        return logits.squeeze(0)  # [vocab_size]
    
    def get_logits_for_position(self, token_position: int, layer_idx: int,
                                data: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """
        Get logits for a specific token position from a specific layer.
        
        Args:
            token_position: Position of the token to analyze
            layer_idx: Layer index (0 to num_layers - 1)
            data: Loaded data dictionary (uses self.loaded_data if None)
            
        Returns:
            Logits tensor of shape [vocab_size]
        """
        if data is None:
            data = self.loaded_data
        
        if data is None:
            raise ValueError("No data loaded. Call load_residual_stream_data() first.")
        
        # Get hidden state from data
        layer_key = f"layer_{layer_idx}"
        hidden_states = data['residual_stream_hidden_states'][layer_key]
        
        if str(token_position) not in hidden_states:
            raise ValueError(f"Token position {token_position} not found in layer {layer_idx}")
        
        # Convert to tensor
        hidden_vector = hidden_states[str(token_position)]
        hidden_tensor = torch.tensor(hidden_vector, dtype=self.model.dtype, device=self.device)
        
        # Apply normalization and unembedding
        logits = self.normalize_and_unembed(hidden_tensor)
        
        return logits
    
    def get_top_k_predictions(self, logits: torch.Tensor, k: int = 10,
                             return_probs: bool = True) -> List[Dict[str, Any]]:
        """
        Get top-k predicted tokens from logits.
        
        Args:
            logits: Logits tensor of shape [vocab_size]
            k: Number of top predictions to return
            return_probs: If True, apply softmax to get probabilities
            
        Returns:
            List of dictionaries with token info, sorted by score (descending)
        """
        # Get top k indices and values
        if return_probs:
            probs = torch.softmax(logits, dim=-1)
            top_values, top_indices = torch.topk(probs, k)
            score_name = "probability"
        else:
            top_values, top_indices = torch.topk(logits, k)
            score_name = "logit"
        
        # Convert to predictions
        predictions = []
        for idx, value in zip(top_indices, top_values):
            token_id = idx.item()
            token_str = self.tokenizer.decode([token_id], skip_special_tokens=False)
            
            predictions.append({
                'rank': len(predictions) + 1,
                'token_id': token_id,
                'token_str': token_str,
                score_name: value.item()
            })
        
        return predictions
    
    def analyze_token_position(self, token_position: int, layer_idx: int,
                               top_k: int = 20, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Perform complete analysis for a specific token position and layer.
        
        Args:
            token_position: Position of the token to analyze
            layer_idx: Layer index (0 to num_layers - 1)
            top_k: Number of top predictions to return
            data: Loaded data dictionary (uses self.loaded_data if None)
            
        Returns:
            Dictionary with analysis results
        """
        if data is None:
            data = self.loaded_data
        
        print(f"\n{'='*80}")
        print(f"ANALYZING TOKEN POSITION {token_position} AT LAYER {layer_idx}")
        print(f"{'='*80}")
        
        # Get the actual token at this position (if available)
        all_tokens = data['token_info']['all_tokens']
        if token_position < len(all_tokens):
            actual_token = all_tokens[token_position]
            print(f"\nActual token at position {token_position}: {repr(actual_token['text'])}")
            print(f"Token ID: {actual_token['id']}")
        else:
            actual_token = None
            print(f"\nPosition {token_position} is beyond saved tokens")
        
        # Get logits
        print(f"\nGetting logits from layer {layer_idx}...")
        logits = self.get_logits_for_position(token_position, layer_idx, data)
        
        # Get top predictions
        print(f"Computing top {top_k} predictions...")
        predictions = self.get_top_k_predictions(logits, k=top_k, return_probs=True)
        
        # Display results
        print(f"\n{'─'*80}")
        print(f"TOP {top_k} PREDICTED TOKENS:")
        print(f"{'─'*80}")
        print(f"{'Rank':<6} {'Probability':<14} {'Token ID':<10} {'Token'}")
        print(f"{'─'*80}")
        
        for pred in predictions:
            print(f"{pred['rank']:<6} {pred['probability']:<14.6f} {pred['token_id']:<10} {repr(pred['token_str'])}")
        
        # Check if actual token is in top k
        if actual_token:
            actual_in_top_k = any(p['token_id'] == actual_token['id'] for p in predictions)
            if actual_in_top_k:
                actual_rank = next(i for i, p in enumerate(predictions, 1) 
                                  if p['token_id'] == actual_token['id'])
                actual_prob = next(p['probability'] for p in predictions 
                                  if p['token_id'] == actual_token['id'])
                print(f"\n✅ Actual token is rank {actual_rank} with probability {actual_prob:.6f}")
            else:
                # Find actual token's rank
                all_probs = torch.softmax(logits, dim=-1)
                actual_prob = all_probs[actual_token['id']].item()
                actual_rank = (all_probs > actual_prob).sum().item() + 1
                print(f"\n⚠️  Actual token is rank {actual_rank} with probability {actual_prob:.6f}")
        
        print(f"{'='*80}\n")
        
        return {
            'token_position': token_position,
            'layer_idx': layer_idx,
            'actual_token': actual_token,
            'predictions': predictions,
            'logits': logits.cpu().numpy().tolist()
        }
    
    def compare_predictions_across_layers(self, token_position: int, 
                                         layer_indices: Optional[List[int]] = None,
                                         top_k: int = 10,
                                         data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Compare predictions for the same token position across different layers.
        
        Args:
            token_position: Position of the token to analyze
            layer_indices: List of layer indices to compare (None = all layers)
            top_k: Number of top predictions to show per layer
            data: Loaded data dictionary (uses self.loaded_data if None)
            
        Returns:
            Dictionary with comparison results
        """
        if data is None:
            data = self.loaded_data
        
        if data is None:
            raise ValueError("No data loaded. Call load_residual_stream_data() first.")
        
        num_layers = data['metadata']['num_layers']
        
        if layer_indices is None:
            layer_indices = list(range(num_layers))
        
        print(f"\n{'='*80}")
        print(f"COMPARING PREDICTIONS ACROSS LAYERS FOR TOKEN POSITION {token_position}")
        print(f"{'='*80}")
        
        # Get actual token
        all_tokens = data['token_info']['all_tokens']
        if token_position < len(all_tokens):
            actual_token = all_tokens[token_position]
            print(f"\nActual token: {repr(actual_token['text'])} (ID: {actual_token['id']})")
        else:
            actual_token = None
        
        results = {
            'token_position': token_position,
            'actual_token': actual_token,
            'layers': {}
        }
        
        # Analyze each layer
        for layer_idx in layer_indices:
            logits = self.get_logits_for_position(token_position, layer_idx, data)
            predictions = self.get_top_k_predictions(logits, k=top_k, return_probs=True)
            
            # Get actual token rank and probability
            if actual_token:
                all_probs = torch.softmax(logits, dim=-1)
                actual_prob = all_probs[actual_token['id']].item()
                actual_rank = (all_probs > actual_prob).sum().item() + 1
            else:
                actual_prob = None
                actual_rank = None
            
            results['layers'][layer_idx] = {
                'predictions': predictions,
                'actual_token_rank': actual_rank,
                'actual_token_prob': actual_prob
            }
        
        # Display summary
        print(f"\n{'─'*80}")
        print(f"LAYER-BY-LAYER SUMMARY:")
        print(f"{'─'*80}")
        print(f"{'Layer':<8} {'Top-1 Token':<25} {'Top-1 Prob':<14} {'Actual Rank':<12} {'Actual Prob'}")
        print(f"{'─'*80}")
        
        for layer_idx in layer_indices:
            layer_data = results['layers'][layer_idx]
            top1 = layer_data['predictions'][0]
            top1_token = top1['token_str'][:20]  # Truncate if too long
            top1_prob = top1['probability']
            
            if actual_token:
                actual_rank = layer_data['actual_token_rank']
                actual_prob = layer_data['actual_token_prob']
                print(f"{layer_idx:<8} {repr(top1_token):<25} {top1_prob:<14.6f} {actual_rank:<12} {actual_prob:.6f}")
            else:
                print(f"{layer_idx:<8} {repr(top1_token):<25} {top1_prob:<14.6f} {'N/A':<12} {'N/A'}")
        
        print(f"{'='*80}\n")
        
        return results
    
    def interactive_analysis(self, json_path: str):
        """
        Run interactive analysis on residual stream data.
        
        Args:
            json_path: Path to residual stream JSON file
        """
        # Load data
        data = self.load_residual_stream_data(json_path)
        
        # Get available positions
        positions = self.get_available_token_positions(data)
        print(f"\n📍 Available token positions: {positions[0]} to {positions[-1]}")
        print(f"   Total positions: {len(positions)}")
        
        # Show some context
        print(f"\n📝 Context:")
        print(f"   Question: {data['question_info']['question'][:80]}...")
        print(f"   Response: {data['response'][:80]}...")
        
        while True:
            print(f"\n{'='*80}")
            print("INTERACTIVE ANALYSIS MENU")
            print(f"{'='*80}")
            print("1. Analyze specific token position at specific layer")
            print("2. Compare predictions across all layers for a token")
            print("3. Compare predictions across selected layers for a token")
            print("4. Show token positions and their actual tokens")
            print("5. Exit")
            print(f"{'='*80}")
            
            choice = input("\nEnter choice (1-5): ").strip()
            
            if choice == "1":
                # Analyze specific position and layer
                pos_input = input(f"Enter token position ({positions[0]}-{positions[-1]}): ").strip()
                try:
                    token_pos = int(pos_input)
                    if token_pos not in positions:
                        print(f"⚠️  Position {token_pos} not available")
                        continue
                except ValueError:
                    print("❌ Invalid input")
                    continue
                
                layer_input = input(f"Enter layer index (0-{data['metadata']['num_layers']-1}): ").strip()
                try:
                    layer_idx = int(layer_input)
                    if layer_idx < 0 or layer_idx >= data['metadata']['num_layers']:
                        print(f"⚠️  Layer {layer_idx} out of range")
                        continue
                except ValueError:
                    print("❌ Invalid input")
                    continue
                
                top_k_input = input("Number of top predictions to show [default: 20]: ").strip()
                top_k = int(top_k_input) if top_k_input else 20
                
                self.analyze_token_position(token_pos, layer_idx, top_k=top_k, data=data)
                
            elif choice == "2":
                # Compare across all layers
                pos_input = input(f"Enter token position ({positions[0]}-{positions[-1]}): ").strip()
                try:
                    token_pos = int(pos_input)
                    if token_pos not in positions:
                        print(f"⚠️  Position {token_pos} not available")
                        continue
                except ValueError:
                    print("❌ Invalid input")
                    continue
                
                top_k_input = input("Number of top predictions per layer [default: 5]: ").strip()
                top_k = int(top_k_input) if top_k_input else 5
                
                self.compare_predictions_across_layers(token_pos, layer_indices=None, 
                                                      top_k=top_k, data=data)
                
            elif choice == "3":
                # Compare across selected layers
                pos_input = input(f"Enter token position ({positions[0]}-{positions[-1]}): ").strip()
                try:
                    token_pos = int(pos_input)
                    if token_pos not in positions:
                        print(f"⚠️  Position {token_pos} not available")
                        continue
                except ValueError:
                    print("❌ Invalid input")
                    continue
                
                layers_input = input(f"Enter layer indices (comma-separated, e.g. '0,10,20,41'): ").strip()
                try:
                    layer_indices = [int(x.strip()) for x in layers_input.split(',')]
                    max_layer = data['metadata']['num_layers'] - 1
                    if any(l < 0 or l > max_layer for l in layer_indices):
                        print(f"⚠️  Some layers out of range (0-{max_layer})")
                        continue
                except ValueError:
                    print("❌ Invalid input")
                    continue
                
                top_k_input = input("Number of top predictions per layer [default: 10]: ").strip()
                top_k = int(top_k_input) if top_k_input else 10
                
                self.compare_predictions_across_layers(token_pos, layer_indices=layer_indices,
                                                      top_k=top_k, data=data)
                
            elif choice == "4":
                # Show token positions
                print(f"\n{'='*80}")
                print("TOKEN POSITIONS")
                print(f"{'='*80}")
                print(f"{'Position':<10} {'Token ID':<10} {'Token'}")
                print(f"{'─'*80}")
                
                for pos in positions[:50]:  # Show first 50
                    token = data['token_info']['all_tokens'][pos]
                    print(f"{pos:<10} {token['id']:<10} {repr(token['text'])}")
                
                if len(positions) > 50:
                    print(f"\n... and {len(positions) - 50} more positions")
                
                print(f"{'='*80}")
                
            elif choice == "5":
                print("\n👋 Exiting...")
                break
                
            else:
                print("❌ Invalid choice")


def main():
    """Main function with CLI interface."""
    import sys
    
    if len(sys.argv) < 2:
        print("\n" + "=" * 80)
        print("Residual Stream Analysis Tool")
        print("=" * 80)
        print("\nUsage:")
        print("  python analyze_residual_stream.py <json_file>")
        print("\nExample:")
        print("  python analyze_residual_stream.py logit_analysis/residual_stream_normal_20250101_120000.json")
        print("\nThis will launch an interactive analysis session.")
        print("=" * 80)
        sys.exit(1)
    
    json_path = sys.argv[1]
    
    if not os.path.exists(json_path):
        print(f"❌ Error: File not found: {json_path}")
        sys.exit(1)
    
    # Initialize analyzer
    analyzer = ResidualStreamAnalyzer(device="auto")
    
    # Load model
    analyzer.load_model()
    
    # Run interactive analysis
    try:
        analyzer.interactive_analysis(json_path)
    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

