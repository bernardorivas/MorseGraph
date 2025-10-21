"""
Test component-specific architecture feature.

This tests:
1. Simple mode (shared architecture) - backward compatibility
2. Advanced mode (component-specific architecture) - new feature
3. Config loading from YAML for both modes
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from MorseGraph.utils import ExperimentConfig
from MorseGraph.config import load_experiment_config


def test_simple_mode():
    """Test that simple mode (shared architecture) works correctly."""
    print("\n" + "="*80)
    print("TEST 1: Simple Mode (Shared Architecture)")
    print("="*80)

    config = ExperimentConfig(
        input_dim=3,
        latent_dim=2,
        hidden_dim=32,
        num_layers=3,
        encoder_activation='tanh',
        decoder_activation='sigmoid',
        latent_dynamics_activation='tanh'
    )

    # Verify simple mode attributes
    assert config.input_dim == 3
    assert config.latent_dim == 2
    assert config.hidden_dim == 32
    assert config.num_layers == 3
    assert config.encoder_activation == 'tanh'
    assert config.decoder_activation == 'sigmoid'
    assert config.latent_dynamics_activation == 'tanh'

    # Verify component-specific attributes are None
    assert config.encoder_hidden_dim is None
    assert config.encoder_num_layers is None
    assert config.decoder_hidden_dim is None
    assert config.decoder_num_layers is None
    assert config.latent_dynamics_hidden_dim is None
    assert config.latent_dynamics_num_layers is None

    print("✓ Simple mode config created successfully")
    print(f"  Shared: hidden_dim={config.hidden_dim}, num_layers={config.num_layers}")

    # Test to_dict
    config_dict = config.to_dict()
    assert 'hidden_dim' in config_dict
    assert 'num_layers' in config_dict
    # Component-specific params should not be in dict if None
    assert 'encoder_hidden_dim' not in config_dict

    print("✓ to_dict works correctly (component-specific params excluded when None)")


def test_advanced_mode():
    """Test that advanced mode (component-specific architecture) works correctly."""
    print("\n" + "="*80)
    print("TEST 2: Advanced Mode (Component-Specific Architecture)")
    print("="*80)

    config = ExperimentConfig(
        input_dim=3,
        latent_dim=2,
        # Also set shared params for backward compatibility
        hidden_dim=32,
        num_layers=3,
        # Component-specific architecture
        encoder_hidden_dim=64,
        encoder_num_layers=4,
        encoder_activation='tanh',
        decoder_hidden_dim=64,
        decoder_num_layers=4,
        decoder_activation='sigmoid',
        latent_dynamics_hidden_dim=32,
        latent_dynamics_num_layers=2,
        latent_dynamics_activation='tanh'
    )

    # Verify component-specific attributes
    assert config.encoder_hidden_dim == 64
    assert config.encoder_num_layers == 4
    assert config.decoder_hidden_dim == 64
    assert config.decoder_num_layers == 4
    assert config.latent_dynamics_hidden_dim == 32
    assert config.latent_dynamics_num_layers == 2

    # Shared params should still be set
    assert config.hidden_dim == 32
    assert config.num_layers == 3

    print("✓ Advanced mode config created successfully")
    print(f"  Encoder: hidden_dim={config.encoder_hidden_dim}, num_layers={config.encoder_num_layers}")
    print(f"  Decoder: hidden_dim={config.decoder_hidden_dim}, num_layers={config.decoder_num_layers}")
    print(f"  Latent Dynamics: hidden_dim={config.latent_dynamics_hidden_dim}, num_layers={config.latent_dynamics_num_layers}")

    # Test to_dict
    config_dict = config.to_dict()
    assert 'encoder_hidden_dim' in config_dict
    assert 'encoder_num_layers' in config_dict
    assert config_dict['encoder_hidden_dim'] == 64
    assert config_dict['encoder_num_layers'] == 4

    print("✓ to_dict works correctly (component-specific params included)")


def test_yaml_loading_simple_mode():
    """Test loading simple mode config from YAML."""
    print("\n" + "="*80)
    print("TEST 3: YAML Loading - Simple Mode (ives_default.yaml)")
    print("="*80)

    config_path = 'examples/configs/ives_default.yaml'
    if not os.path.exists(config_path):
        print(f"⚠ Skipping: {config_path} not found")
        return

    config = load_experiment_config(config_path, verbose=False)

    # Should have shared params
    assert hasattr(config, 'hidden_dim')
    assert hasattr(config, 'num_layers')
    assert config.hidden_dim == 32
    assert config.num_layers == 3

    # Component-specific params should be None (simple mode)
    assert config.encoder_hidden_dim is None
    assert config.encoder_num_layers is None

    print("✓ Simple mode YAML loaded successfully")
    print(f"  Shared: hidden_dim={config.hidden_dim}, num_layers={config.num_layers}")
    print(f"  Activations: encoder={config.encoder_activation}, decoder={config.decoder_activation}, dynamics={config.latent_dynamics_activation}")


def test_yaml_loading_advanced_mode():
    """Test loading advanced mode config from YAML."""
    print("\n" + "="*80)
    print("TEST 4: YAML Loading - Advanced Mode (ives_morals.yaml)")
    print("="*80)

    config_path = 'examples/configs/ives_morals.yaml'
    if not os.path.exists(config_path):
        print(f"⚠ Skipping: {config_path} not found")
        return

    config = load_experiment_config(config_path, verbose=False)

    # Should have component-specific params
    assert config.encoder_hidden_dim == 32
    assert config.encoder_num_layers == 1
    assert config.decoder_hidden_dim == 32
    assert config.decoder_num_layers == 1
    assert config.latent_dynamics_hidden_dim == 32
    assert config.latent_dynamics_num_layers == 1

    # Check activations
    assert config.encoder_activation == 'tanh'
    assert config.decoder_activation == 'sigmoid'
    assert config.latent_dynamics_activation == 'tanh'

    print("✓ Advanced mode YAML loaded successfully")
    print(f"  Encoder: hidden_dim={config.encoder_hidden_dim}, num_layers={config.encoder_num_layers}, activation={config.encoder_activation}")
    print(f"  Decoder: hidden_dim={config.decoder_hidden_dim}, num_layers={config.decoder_num_layers}, activation={config.decoder_activation}")
    print(f"  Latent Dynamics: hidden_dim={config.latent_dynamics_hidden_dim}, num_layers={config.latent_dynamics_num_layers}, activation={config.latent_dynamics_activation}")


def test_model_instantiation():
    """Test that models can be instantiated with component-specific parameters."""
    print("\n" + "="*80)
    print("TEST 5: Model Instantiation with Component-Specific Architecture")
    print("="*80)

    try:
        import torch
        from MorseGraph.models import Encoder, Decoder, LatentDynamics

        # Test with component-specific params
        encoder = Encoder(
            input_dim=3,
            latent_dim=2,
            hidden_dim=64,
            num_layers=4,
            output_activation='tanh'
        )

        decoder = Decoder(
            latent_dim=2,
            output_dim=3,
            hidden_dim=64,
            num_layers=4,
            output_activation='sigmoid'
        )

        latent_dynamics = LatentDynamics(
            latent_dim=2,
            hidden_dim=32,
            num_layers=2,
            output_activation='tanh'
        )

        print("✓ Models instantiated successfully")
        print(f"  Encoder: {sum(p.numel() for p in encoder.parameters()):,} parameters")
        print(f"  Decoder: {sum(p.numel() for p in decoder.parameters()):,} parameters")
        print(f"  Latent Dynamics: {sum(p.numel() for p in latent_dynamics.parameters()):,} parameters")

        # Test forward pass
        x = torch.randn(10, 3)
        z = encoder(x)
        x_recon = decoder(z)
        z_next = latent_dynamics(z)

        assert z.shape == (10, 2)
        assert x_recon.shape == (10, 3)
        assert z_next.shape == (10, 2)

        print("✓ Forward pass works correctly")

    except ImportError:
        print("⚠ PyTorch not available, skipping model instantiation test")


if __name__ == '__main__':
    print("\n" + "="*80)
    print("COMPONENT-SPECIFIC ARCHITECTURE FEATURE TESTS")
    print("="*80)

    try:
        test_simple_mode()
        test_advanced_mode()
        test_yaml_loading_simple_mode()
        test_yaml_loading_advanced_mode()
        test_model_instantiation()

        print("\n" + "="*80)
        print("ALL TESTS PASSED! ✓")
        print("="*80)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
