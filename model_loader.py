"""
Model configuration loader for KV cache sizing app.
Loads model configs from aiconfigurator and calculates KV cache parameters.
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional


class ModelConfig:
    """Represents a model configuration with KV cache parameters."""

    def __init__(self, name: str, config: dict):
        self.name = name
        self.config = config

        # Extract architecture parameters
        self.num_layers = config.get('num_hidden_layers', config.get('num_decoder_layers', 32))
        self.hidden_size = config.get('hidden_size', 4096)
        self.num_attention_heads = config.get('num_attention_heads', 32)
        self.num_key_value_heads = config.get('num_key_value_heads', self.num_attention_heads)

        
        # Get head dimension from config, or calculate if not present
        self.head_dim = config.get('head_dim', self.hidden_size // self.num_attention_heads)

    def get_kv_bytes_per_token(self, dtype: str = 'fp16') -> int:
        """
        Calculate KV cache bytes per token.

        Formula: 2 * num_layers * head_dim * num_kv_heads * bytes_per_param

        Args:
            dtype: Data type ('fp16', 'bf16', 'fp8', 'fp32')

        Returns:
            Bytes per token for KV cache
        """
        bytes_per_param = {
            'fp32': 4,
            'fp16': 2,
            'bf16': 2,
            'fp8': 1,
        }

        dtype_lower = dtype.lower()
        if dtype_lower not in bytes_per_param:
            raise ValueError(f"Unknown dtype: {dtype}. Supported: {list(bytes_per_param.keys())}")

        bytes_per = bytes_per_param[dtype_lower]

        # 2 (K and V) * layers * head_dim * num_kv_heads * bytes_per_param
        kv_bytes = 2 * self.num_layers * self.head_dim * self.num_key_value_heads * bytes_per

        return kv_bytes

    def __repr__(self):
        return (f"ModelConfig(name={self.name}, layers={self.num_layers}, "
                f"hidden={self.hidden_size}, heads={self.num_attention_heads}, "
                f"kv_heads={self.num_key_value_heads}, head_dim={self.head_dim})")


class ModelLoader:
    """Loads and manages model configurations from aiconfigurator."""

    def __init__(self, config_dir: Optional[str] = None):
        # Try multiple possible locations for model configs
        possible_paths = [
            config_dir,  # User-specified path
            "./model_configs",  # Local directory (for deployment)
            "/home/katyag/aiconfigurator/src/aiconfigurator/model_configs",  # Original path
            str(Path(__file__).parent / "model_configs"),  # Relative to this file
        ]

        self.config_dir = None
        for path in possible_paths:
            if path and Path(path).exists():
                self.config_dir = Path(path)
                break

        if self.config_dir is None:
            raise ValueError(
                f"Model config directory not found. Tried:\n" +
                "\n".join([f"  - {p}" for p in possible_paths if p])
            )

        self.models: Dict[str, ModelConfig] = {}
        self._load_models()

    def _load_models(self):
        """Load all model configs from the directory."""
        if not self.config_dir.exists():
            raise ValueError(f"Model config directory not found: {self.config_dir}")

        for config_file in self.config_dir.glob("*_config.json"):
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)

                # Extract model name from filename (remove _config.json)
                model_name = config_file.stem.replace('_config', '')

                # Create ModelConfig
                model_config = ModelConfig(model_name, config)
                self.models[model_name] = model_config

            except Exception as e:
                print(f"Warning: Failed to load {config_file.name}: {e}")

    def get_model(self, name: str) -> Optional[ModelConfig]:
        """Get a model config by name."""
        return self.models.get(name)

    def list_models(self) -> list:
        """List all available model names."""
        return sorted(self.models.keys())

    def get_model_summary(self, name: str, dtype: str = 'fp16') -> dict:
        """Get a summary of model parameters including KV cache size."""
        model = self.get_model(name)
        if not model:
            return {}

        kv_bytes = model.get_kv_bytes_per_token(dtype)

        return {
            'name': model.name,
            'num_layers': model.num_layers,
            'hidden_size': model.hidden_size,
            'num_attention_heads': model.num_attention_heads,
            'num_key_value_heads': model.num_key_value_heads,
            'head_dim': model.head_dim,
            'kv_bytes_per_token': kv_bytes,
            'kv_kb_per_token': kv_bytes / 1024,
        }


# Create a global instance for easy access
model_loader = ModelLoader()


if __name__ == "__main__":
    # Test the loader
    loader = ModelLoader()
    print(f"Loaded {len(loader.models)} models\n")

    # Test with a few models
    test_models = ['meta-llama--Meta-Llama-3.1-8B', 'meta-llama--Meta-Llama-3.1-70B']

    for model_name in test_models:
        if model_name in loader.models:
            print(f"\n{model_name}:")
            for dtype in ['fp16', 'fp8']:
                summary = loader.get_model_summary(model_name, dtype)
                print(f"  {dtype}: {summary['kv_bytes_per_token']:,} bytes/token "
                      f"({summary['kv_kb_per_token']:.1f} KB/token)")
