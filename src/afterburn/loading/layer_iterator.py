"""Memory-efficient layer-by-layer model weight iteration."""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

import torch
from safetensors import safe_open

from afterburn.loading.checkpoint import CheckpointInfo, detect_checkpoint

logger = logging.getLogger(__name__)

# Pattern to extract layer index from parameter names
# Matches patterns like: model.layers.0.self_attn.q_proj.weight
LAYER_INDEX_PATTERN = re.compile(r"(?:model\.layers|transformer\.h|gpt_neox\.layers)\.(\d+)\.")


class LayerIterator:
    """Memory-efficient iterator over model layers.

    Uses safetensors memory-mapping or PyTorch lazy loading to iterate
    over model weights one layer at a time, never loading the full model.
    """

    def __init__(
        self,
        model_id: str,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        self.model_id = model_id
        self.device = device if device is not None else torch.device("cpu")
        self.dtype = dtype
        self._checkpoint = detect_checkpoint(model_id)
        self._layer_map = self._build_layer_map()

    @property
    def checkpoint(self) -> CheckpointInfo:
        return self._checkpoint

    @property
    def num_layers(self) -> int:
        return self._checkpoint.num_hidden_layers or 0

    @property
    def layer_names(self) -> list[str]:
        return sorted(self._layer_map.keys())

    def _build_layer_map(self) -> dict[str, list[str]]:
        """Build a mapping from logical layer names to parameter names.

        Groups parameters by their layer (e.g., 'layer_0', 'layer_1', ...)
        and also identifies non-layer parameters (embeddings, final norm, lm_head).
        """
        all_keys = self._get_all_parameter_keys()
        layer_map: dict[str, list[str]] = defaultdict(list)

        for key in all_keys:
            match = LAYER_INDEX_PATTERN.search(key)
            if match:
                layer_idx = int(match.group(1))
                layer_map[f"layer_{layer_idx}"].append(key)
            elif any(tok in key for tok in ["embed", "wte", "wpe"]):
                layer_map["embedding"].append(key)
            elif any(tok in key for tok in ["ln_f", "norm", "final_layer_norm", "model.norm"]):
                if "layers." not in key:
                    layer_map["final_norm"].append(key)
                else:
                    # This is a per-layer norm, extract layer index
                    match2 = re.search(r"\.(\d+)\.", key)
                    if match2:
                        layer_idx = int(match2.group(1))
                        layer_map[f"layer_{layer_idx}"].append(key)
                    else:
                        layer_map["other"].append(key)
            elif any(tok in key for tok in ["lm_head", "output"]):
                layer_map["lm_head"].append(key)
            else:
                layer_map["other"].append(key)

        return dict(layer_map)

    def _get_all_parameter_keys(self) -> list[str]:
        """Get all parameter keys from the checkpoint files."""
        if self._checkpoint.format == "safetensors":
            return self._get_safetensors_keys()
        else:
            return self._get_pytorch_keys()

    def _get_safetensors_keys(self) -> list[str]:
        """Get parameter keys from safetensors files."""
        keys = []
        for wf in self._checkpoint.weight_files:
            with safe_open(str(wf), framework="pt", device="cpu") as f:  # type: ignore[no-untyped-call]
                keys.extend(f.keys())
        return keys

    def _get_pytorch_keys(self) -> list[str]:
        """Get parameter keys from PyTorch bin files."""
        keys = []
        for wf in self._checkpoint.weight_files:
            # Use weights_only=True for security
            state_dict = torch.load(str(wf), map_location="cpu", weights_only=True)
            keys.extend(state_dict.keys())
            del state_dict
        return keys

    def get_layer(self, layer_name: str) -> dict[str, torch.Tensor]:
        """Load a specific layer by name.

        Returns a dict mapping parameter names to tensors.
        Tensors are loaded to the configured device and dtype.
        """
        if layer_name not in self._layer_map:
            raise KeyError(f"Layer '{layer_name}' not found. Available: {self.layer_names}")

        param_keys = self._layer_map[layer_name]
        return self._load_params(param_keys)

    def _load_params(self, param_keys: list[str]) -> dict[str, torch.Tensor]:
        """Load specific parameters from checkpoint files."""
        params = {}

        if self._checkpoint.format == "safetensors":
            # Build index: which file contains which key
            key_to_file = self._build_safetensors_index()
            for key in param_keys:
                if key not in key_to_file:
                    logger.warning("Parameter '%s' not found in any weight file", key)
                    continue
                wf = key_to_file[key]
                with safe_open(str(wf), framework="pt", device=str(self.device)) as f:  # type: ignore[no-untyped-call]
                    tensor = f.get_tensor(key)
                    if tensor.dtype != self.dtype:
                        tensor = tensor.to(self.dtype)
                    params[key] = tensor
        else:
            # PyTorch format: load full state dict and extract
            for wf in self._checkpoint.weight_files:
                state_dict = torch.load(str(wf), map_location=str(self.device), weights_only=True)
                for key in param_keys:
                    if key in state_dict:
                        tensor = state_dict[key]
                        if tensor.dtype != self.dtype:
                            tensor = tensor.to(self.dtype)
                        params[key] = tensor
                del state_dict

        return params

    def _build_safetensors_index(self) -> dict[str, Path]:
        """Build an index mapping parameter keys to their safetensors files."""
        # Check for index file first
        index_path = self._checkpoint.local_path / "model.safetensors.index.json"
        if index_path.exists():
            with open(index_path) as f:
                index_data = json.load(f)
            weight_map = index_data.get("weight_map", {})
            return {
                k: self._checkpoint.local_path / v for k, v in weight_map.items()
            }

        # No index file — scan all files
        key_to_file: dict[str, Path] = {}
        for wf in self._checkpoint.weight_files:
            with safe_open(str(wf), framework="pt", device="cpu") as f:  # type: ignore[no-untyped-call]
                for key in f.keys():  # type: ignore[attr-defined]  # noqa: SIM118
                    key_to_file[key] = wf
        return key_to_file

    def iterate_layers(self) -> Iterator[tuple[str, dict[str, torch.Tensor]]]:
        """Yield (layer_name, {param_name: tensor}) one layer at a time.

        Layers are yielded in order: embedding, layer_0..layer_N, final_norm, lm_head.
        """
        # Yield in logical order
        order = []
        if "embedding" in self._layer_map:
            order.append("embedding")

        for i in range(self.num_layers):
            name = f"layer_{i}"
            if name in self._layer_map:
                order.append(name)

        if "final_norm" in self._layer_map:
            order.append("final_norm")
        if "lm_head" in self._layer_map:
            order.append("lm_head")
        if "other" in self._layer_map:
            order.append("other")

        for layer_name in order:
            params = self.get_layer(layer_name)
            yield layer_name, params
            # Free memory after yielding
            del params
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @staticmethod
    def paired_iterate(
        base_iter: LayerIterator,
        trained_iter: LayerIterator,
    ) -> Iterator[tuple[str, dict[str, torch.Tensor], dict[str, torch.Tensor]]]:
        """Iterate over matching layers from both models simultaneously.

        Yields (layer_name, base_params, trained_params) for each layer.
        Only yields layers that exist in both models.
        """
        base_layers = set(base_iter.layer_names)
        trained_layers = set(trained_iter.layer_names)
        common_layers = base_layers & trained_layers

        if base_layers != trained_layers:
            only_base = base_layers - trained_layers
            only_trained = trained_layers - base_layers
            if only_base:
                logger.warning("Layers only in base model: %s", only_base)
            if only_trained:
                logger.warning("Layers only in trained model: %s", only_trained)

        # Iterate in order
        order = []
        if "embedding" in common_layers:
            order.append("embedding")

        max_layers = max(base_iter.num_layers, trained_iter.num_layers)
        for i in range(max_layers):
            name = f"layer_{i}"
            if name in common_layers:
                order.append(name)

        for special in ["final_norm", "lm_head", "other"]:
            if special in common_layers:
                order.append(special)

        for layer_name in order:
            base_params = base_iter.get_layer(layer_name)
            trained_params = trained_iter.get_layer(layer_name)
            yield layer_name, base_params, trained_params
            del base_params, trained_params
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
