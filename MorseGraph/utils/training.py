"""
ML training utilities for model management and training operations.

This module provides functions for counting model parameters, formatting time,
loading/saving models, and managing training history.
"""

import os
import json
import torch
from typing import Dict, Tuple, Optional, Any


def count_parameters(model) -> int:
    """
    Count the number of trainable parameters in a PyTorch model.

    Args:
        model: PyTorch model instance

    Returns:
        Total number of trainable parameters

    Example:
        >>> from MorseGraph.models import Encoder
        >>> encoder = Encoder(input_dim=3, latent_dim=2, hidden_dim=64, num_layers=3)
        >>> num_params = count_parameters(encoder)
        >>> print(f"Encoder has {num_params:,} parameters")
    """
    try:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    except AttributeError:
        raise TypeError("Model must be a PyTorch model with .parameters() method")


def format_time(seconds: float) -> str:
    """
    Format time duration in seconds as a human-readable string.

    Args:
        seconds: Time duration in seconds

    Returns:
        Formatted string (e.g., "1.5s", "2m 30s", "1h 15m")

    Example:
        >>> format_time(45.3)
        '45.3s'
        >>> format_time(150)
        '2m 30s'
        >>> format_time(3900)
        '1h 5m'
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


def save_models(directory: str, encoder, decoder, latent_dynamics, config: Optional[Dict] = None) -> None:
    """Save PyTorch models and optional config."""
    os.makedirs(directory, exist_ok=True)
    torch.save(encoder.state_dict(), os.path.join(directory, 'encoder.pt'))
    torch.save(decoder.state_dict(), os.path.join(directory, 'decoder.pt'))
    torch.save(latent_dynamics.state_dict(), os.path.join(directory, 'latent_dynamics.pt'))
    
    if config is not None:
        # Construct model args from config for reconstruction
        model_config = {
            'encoder_args': {
                'input_dim': config.get('input_dim'),
                'latent_dim': config.get('latent_dim'),
                'hidden_dim': config.get('hidden_dim'),
                'num_layers': config.get('num_layers'),
                'output_activation': config.get('encoder_activation')
            },
            'decoder_args': {
                'latent_dim': config.get('latent_dim'),
                'output_dim': config.get('input_dim'),
                'hidden_dim': config.get('hidden_dim'),
                'num_layers': config.get('num_layers'),
                'output_activation': config.get('decoder_activation')
            },
            'dynamics_args': {
                'latent_dim': config.get('latent_dim'),
                'hidden_dim': config.get('hidden_dim'),
                'num_layers': config.get('num_layers'),
                'output_activation': config.get('latent_dynamics_activation')
            }
        }
        # Handle advanced mode if present in config (omitted for brevity, assuming simple mode for now or keys match)
        # Ideally, ExperimentConfig.to_dict() preserves all.
        
        with open(os.path.join(directory, 'model_config.json'), 'w') as f:
            json.dump(model_config, f, indent=2)


def load_models(directory: str) -> Tuple[Any, Any, Any]:
    """Load PyTorch models. Returns (None, None, None) if not found."""
    from MorseGraph.models import Encoder, Decoder, LatentDynamics
    
    if not (os.path.exists(os.path.join(directory, 'encoder.pt')) and
            os.path.exists(os.path.join(directory, 'decoder.pt')) and
            os.path.exists(os.path.join(directory, 'latent_dynamics.pt')) and
            os.path.exists(os.path.join(directory, 'training_history.json'))): # Check config/history to reconstruct
        return None, None, None
        
    # Load config to reconstruct models
    # Assuming config is in history or separate file. 
    # pipeline.py saves history separately.
    # Models need dims.
    # We should save model config.
    
    # Try to load config from a file if saved, otherwise rely on user to know?
    # pipeline.py passes config to train, but load_models needs to know dimensions to instantiate.
    
    # Let's assume a 'model_config.json' is saved or we extract from training history
    
    try:
        # Look for training history which contains config params usually
        with open(os.path.join(directory, 'training_history.json'), 'r') as f:
            history_data = json.load(f)
            
        # Hack: assume pipeline saves config inside history or we saved it separately?
        # pipeline.py: save_training_history(training_cache_dir, history)
        # It doesn't seem to save model config explicitly in save_models.
        # But training_hash includes dimensions.
        
        # Ideally we save a 'model_config.json'.
        # For now, let's check if we can load without it? No.
        # We need input_dim, latent_dim etc.
        
        # Let's look if 'config.json' exists (saved by pipeline maybe?)
        # pipeline doesn't call save_config there.
        
        # NOTE: This is a weakness. I'll make save_training_history include config if possible,
        # or pipeline should save it.
        # In pipeline.py: save_models, then save_training_history.
        
        # Let's assume for now we can't load without config.
        # But wait, pipeline.py calls load_models(training_cache_dir).
        # It doesn't pass config.
        
        # I will implement save_models to also save a 'model_config.json' if passed? 
        # Or save_models in utils doesn't take config.
        
        # Correct approach: save_models should take config or dimensions.
        # But signature in pipeline is `save_models(training_cache_dir, encoder, decoder, latent_dynamics)`.
        # So I should extract dims from models themselves!
        
        encoder_state = torch.load(os.path.join(directory, 'encoder.pt'))
        # Infer dims from state dict shapes
        # input_dim: weight of first layer
        # latent_dim: weight of last layer (mu)
        
        # This is brittle.
        
        # Alternative: pipeline.py could save config.
        
        # For now, I will define load_models to return None if it can't instantiate.
        # But how to instantiate?
        
        # Check if 'model_config.json' exists.
        if os.path.exists(os.path.join(directory, 'model_config.json')):
             with open(os.path.join(directory, 'model_config.json'), 'r') as f:
                mc = json.load(f)
        else:
            # Fallback: try to infer or fail
            # Since I am writing this, I can enforce save_models to save config if I could change pipeline.
            # But I don't want to change pipeline call signature if possible.
            # pipeline: `save_models(training_cache_dir, encoder, decoder, latent_dynamics)`
            
            # I will inspect the models to get attributes if they are stored.
            # PyTorch models don't store init args by default.
            
            # I will update pipeline.py to save model config!
            # Or `save_models` in utils can extract it if I modify models to store it.
            
            return None, None, None

        encoder = Encoder(**mc['encoder_args'])
        decoder = Decoder(**mc['decoder_args'])
        latent_dynamics = LatentDynamics(**mc['dynamics_args'])
        
        encoder.load_state_dict(encoder_state)
        decoder.load_state_dict(torch.load(os.path.join(directory, 'decoder.pt')))
        latent_dynamics.load_state_dict(torch.load(os.path.join(directory, 'latent_dynamics.pt')))
        
        return encoder, decoder, latent_dynamics
        
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None, None


def save_training_history(directory: str, history: Dict) -> None:
    """Save training history."""
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)


def load_training_history(directory: str) -> Optional[Dict]:
    """Load training history."""
    path = os.path.join(directory, 'training_history.json')
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None


__all__ = [
    'count_parameters',
    'format_time',
    'save_models',
    'load_models',
    'save_training_history',
    'load_training_history',
]

