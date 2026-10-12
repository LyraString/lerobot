# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Native LoRA adapted from sen-code-lost/openpi's PyTorch implementation.

Reference: https://github.com/Physical-Intelligence/openpi/issues/842
Adapters inherit the base weight's dtype. Fully trained modules are unfrozen in
place, and the standard policy checkpoint contains both base and adapter weights.
"""

import logging

import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from .configuration_pi05 import PI05Config, PI05NativeLoRAConfig

_TARGET_MODULES = frozenset({"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"})
_FULL_TRAINING_MODULES = frozenset(
    {"action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out", "proprio_history_proj"}
)


class NativeLoRALinear(nn.Module):
    """Preserve a Linear's weight names and storage while adding A/B adapters."""

    def __init__(self, linear: nn.Linear, config: PI05NativeLoRAConfig):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = linear.weight
        self.bias = linear.bias
        self.scaling = config.alpha / config.rank
        self.dropout = nn.Dropout(config.dropout) if config.dropout else nn.Identity()
        self.lora_a = nn.Parameter(linear.weight.new_empty(config.rank, linear.in_features))
        self.lora_b = nn.Parameter(linear.weight.new_zeros(linear.out_features, config.rank))
        nn.init.normal_(self.lora_a, std=config.init_std)

    def forward(self, x: Tensor) -> Tensor:
        result = F.linear(x, self.weight, self.bias)
        adapter = F.linear(F.linear(self.dropout(x), self.lora_a), self.lora_b)
        return result + adapter * self.scaling


def apply_native_lora(model: nn.Module, config: PI05Config) -> None:
    """Train Transformer adapters, vision (when enabled), and action/time projections."""
    if config.native_lora is None:
        return
    model.requires_grad_(False)
    count = 0
    for name, module in list(model.named_modules()):
        parts = name.split(".")
        is_vlm = "language_model" in parts and not config.train_expert_only
        is_expert = "gemma_expert" in parts
        if not (is_vlm or is_expert) or parts[-1] not in _TARGET_MODULES:
            continue
        if not isinstance(module, nn.Linear):
            continue
        parent_name, attr = name.rsplit(".", 1)
        setattr(model.get_submodule(parent_name), attr, NativeLoRALinear(module, config.native_lora))
        count += 1
    if not count:
        raise ValueError("PI05 native LoRA found no Transformer linear layers")

    train_vision = not config.freeze_vision_encoder and not config.train_expert_only
    for name, parameter in model.named_parameters():
        parts = name.split(".")
        if (
            parts[-1] in {"lora_a", "lora_b"}
            or bool(_FULL_TRAINING_MODULES.intersection(parts))
            or (train_vision and "vision_tower" in parts)
        ):
            parameter.requires_grad_(True)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info("PI05 native LoRA: %s Transformer linear layers, %s trainable parameters", count, trainable)
