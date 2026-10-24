# Copyright 2026 LeRobot contributors. All rights reserved.
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

import copy

import pytest
import torch
import torch.distributed as dist

from lerobot.configs.accelerator import AcceleratorConfig, GradientAccumulationConfig
from lerobot.configs.parallelism import ParallelismConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.distributed.factory import set_fsdp_wrap_modules
from lerobot.policies.pi05 import modeling_pi05
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.pi_gemma import PiGemmaRMSNorm
from tests.distributed.test_multigpu_training import _init_worker_env, _spawn


def _make_tiny_policy():
    config_mapping = modeling_pi05.CONFIG_MAPPING
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            modeling_pi05,
            "CONFIG_MAPPING",
            {
                "paligemma": lambda: config_mapping["paligemma"](
                    vision_config={
                        "hidden_size": 32,
                        "num_hidden_layers": 1,
                        "num_attention_heads": 4,
                        "patch_size": 8,
                    }
                ),
                "gemma": config_mapping["gemma"],
            },
        )
        patch.setattr(
            modeling_pi05,
            "get_gemma_config",
            lambda variant: modeling_pi05.GemmaConfig(
                width=32 if variant == "gemma_2b" else 16,
                depth=2,
                mlp_dim=64,
                num_heads=8,
                num_kv_heads=1,
                head_dim=4,
            ),
        )
        policy = PI05Policy(
            PI05Config(
                device="cpu",
                dtype="float32",
                image_resolution=(16, 16),
                chunk_size=4,
                n_action_steps=4,
                gradient_checkpointing=True,
                input_features={
                    "observation.images.exterior": PolicyFeature(FeatureType.VISUAL, (3, 16, 16)),
                    "observation.state": PolicyFeature(FeatureType.STATE, (14,)),
                },
                output_features={"action": PolicyFeature(FeatureType.ACTION, (14,))},
            )
        )
    projector = policy.model.paligemma_with_expert.paligemma.model.multi_modal_projector
    projector.linear = torch.nn.Linear(32, 32)
    # Open the initially-zero adaptive gates so gradients also reach vision and the VLM.
    with torch.no_grad():
        for module in policy.modules():
            if isinstance(module, PiGemmaRMSNorm) and module.dense is not None:
                module.dense.weight.normal_(std=0.02)
                module.dense.bias.normal_(std=0.02)
    return policy


def _parity_worker(rank, world_size, port, mixed_precision):
    _init_worker_env(rank, world_size, port)
    torch.manual_seed(42)
    parallelism = ParallelismConfig(dp_shard=world_size)
    parallelism.resolve(world_size)
    accelerator = AcceleratorConfig(
        mixed_precision=mixed_precision,
        gradient_accumulation=GradientAccumulationConfig(steps=2),
    ).build(parallelism)
    policy = _make_tiny_policy()
    reference = copy.deepcopy(policy).to(accelerator.device)
    batch = {
        "observation.images.exterior": torch.rand(1, 3, 16, 16, device=accelerator.device),
        "observation.language.tokens": torch.tensor([[2, 3, 4, 5]], device=accelerator.device),
        "observation.language.attention_mask": torch.ones(1, 4, dtype=torch.bool, device=accelerator.device),
        "action": torch.randn(1, 4, 14, device=accelerator.device) + rank,
    }
    reference_batch = batch
    if mixed_precision == "bf16":
        reference.bfloat16()
        reference_batch = {
            name: value.bfloat16() if value.is_floating_point() else value for name, value in batch.items()
        }
    for micro_step in range(2):
        torch.manual_seed(100 + rank * 2 + micro_step)
        with accelerator.autocast():
            loss, _ = reference(reference_batch)
        (loss / 2).backward()
    for parameter in reference.parameters():
        if parameter.grad is not None:
            dist.all_reduce(parameter.grad)
            parameter.grad.div_(world_size)

    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4)
    set_fsdp_wrap_modules(accelerator, policy)
    policy, optimizer = accelerator.prepare(policy, optimizer)
    for micro_step in range(2):
        torch.manual_seed(100 + rank * 2 + micro_step)
        with accelerator.accumulate(policy):
            with accelerator.autocast():
                loss, _ = policy(batch)
            accelerator.backward(loss)
    expected = dict(reference.named_parameters())
    for name, parameter in policy.named_parameters():
        if expected[name].grad is None:
            continue
        assert parameter.dtype == torch.float32
        gradient = parameter.grad.full_tensor()
        torch.testing.assert_close(
            gradient, expected[name].grad.float(), rtol=0.1 if mixed_precision == "bf16" else 1e-4, atol=3e-5
        )
    optimizer.step()
    accelerator.end_training()


@pytest.mark.multigpu
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA GPUs")
@pytest.mark.parametrize("mixed_precision", ["no", "bf16"])
def test_pi05_fsdp_joint_attention_gradients(mixed_precision):
    pytest.importorskip("transformers")
    pytest.importorskip("accelerate")
    _spawn(2, _parity_worker, mixed_precision)
