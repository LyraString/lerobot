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

from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file
from torch import nn

from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.pi05.configuration_pi05 import PI05Config, PI05NativeLoRAConfig
from lerobot.policies.pi05.lora import NativeLoRALinear, apply_native_lora
from lerobot.policies.pi05.modeling_pi05 import PaliGemmaWithExpertModel, PI05Policy


def test_native_lora_preserves_base_storage_and_initial_output():
    linear = nn.Linear(8, 6, dtype=torch.bfloat16)
    adapter = NativeLoRALinear(linear, PI05NativeLoRAConfig(rank=4, alpha=4))
    x = torch.randn(2, 3, 8, dtype=torch.bfloat16)
    assert adapter.weight is linear.weight
    assert adapter.bias is linear.bias
    assert adapter.lora_a.dtype == torch.bfloat16
    assert adapter.lora_b.dtype == torch.bfloat16
    torch.testing.assert_close(adapter(x), linear(x), rtol=0, atol=0)
    adapter(x).float().square().mean().backward()
    assert adapter.lora_b.grad.abs().sum() > 0


def _tiny_model():
    model = nn.Module()
    model.paligemma_with_expert = nn.Module()
    model.paligemma_with_expert.paligemma = nn.Module()
    model.paligemma_with_expert.paligemma.language_model = nn.Sequential()
    model.paligemma_with_expert.paligemma.language_model.add_module("q_proj", nn.Linear(8, 8))
    model.paligemma_with_expert.paligemma.vision_tower = nn.Linear(8, 8)
    model.paligemma_with_expert.gemma_expert = nn.Sequential()
    model.paligemma_with_expert.gemma_expert.add_module("up_proj", nn.Linear(8, 8))
    model.action_in_proj = nn.Linear(8, 8)
    model.action_out_proj = nn.Linear(8, 8)
    model.time_mlp_in = nn.Linear(8, 8)
    model.time_mlp_out = nn.Linear(8, 8)
    return model


@pytest.mark.parametrize("freeze_vision,expert_only", [(False, False), (True, False), (False, True)])
def test_native_lora_updates_the_intended_modules(freeze_vision, expert_only):
    model = _tiny_model()
    vision = model.paligemma_with_expert.paligemma.vision_tower
    config = PI05Config(
        native_lora=PI05NativeLoRAConfig(rank=4, alpha=4),
        freeze_vision_encoder=freeze_vision,
        train_expert_only=expert_only,
    )
    apply_native_lora(model, config)
    # Unfreezing vision must not add the PEFT modules_to_save copy.
    assert model.paligemma_with_expert.paligemma.vision_tower is vision
    assert vision.weight.requires_grad == (not freeze_vision and not expert_only)
    expert = model.paligemma_with_expert.gemma_expert.up_proj
    assert not expert.weight.requires_grad
    assert expert.lora_a.requires_grad and expert.lora_b.requires_grad
    vlm = model.paligemma_with_expert.paligemma.language_model.q_proj
    assert isinstance(vlm, NativeLoRALinear) == (not expert_only)

    before = {key: p.detach().clone() for key, p in model.named_parameters()}
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-2)
    x = torch.randn(2, 8)
    for _ in range(2):
        optimizer.zero_grad()
        loss = (model.action_out_proj(expert(vlm(vision(x)))) - 1).square().mean()
        loss.backward()
        optimizer.step()
    after = dict(model.named_parameters())
    for key, parameter in after.items():
        if not parameter.requires_grad:
            assert torch.equal(parameter, before[key])
    assert not torch.equal(expert.lora_a, before["paligemma_with_expert.gemma_expert.up_proj.lora_a"])
    assert not torch.equal(expert.lora_b, before["paligemma_with_expert.gemma_expert.up_proj.lora_b"])
    assert not torch.equal(model.action_out_proj.weight, before["action_out_proj.weight"])
    assert torch.equal(vision.weight, before["paligemma_with_expert.paligemma.vision_tower.weight"]) == (
        freeze_vision or expert_only
    )


@pytest.fixture
def tiny_policy(monkeypatch):
    # Exercise the real policy checkpoint API without constructing the 4B model.
    from lerobot.policies.pi05 import modeling_pi05

    class TinyPI05(nn.Module):
        def __init__(self, config, rtc_processor=None):
            super().__init__()
            self.core = _tiny_model()
            apply_native_lora(self.core, config)

    monkeypatch.setattr(modeling_pi05, "PI05Pytorch", TinyPI05)
    return PI05Policy


def test_native_lora_base_load_save_reload_and_missing_adapter(tiny_policy, tmp_path):
    base_dir = tmp_path / "base"
    config = PI05Config(device="cpu", push_to_hub=False)
    tiny_policy(config).save_pretrained(base_dir)
    native_config = PI05Config(
        device="cpu", push_to_hub=False, native_lora=PI05NativeLoRAConfig(rank=4, alpha=4)
    )
    native = tiny_policy.from_pretrained(base_dir, config=native_config, local_files_only=True)
    # A zero-initialized adapter loads the original base weights exactly.
    with torch.no_grad():
        for name, parameter in native.named_parameters():
            if name.endswith(".lora_b"):
                parameter.fill_(0.1)
    native_dir = tmp_path / "native"
    native.save_pretrained(native_dir)
    reloaded = tiny_policy.from_pretrained(native_dir, local_files_only=True)
    assert reloaded.config.native_lora.rank == 4
    assert not reloaded.config.use_peft
    for key, tensor in native.state_dict().items():
        torch.testing.assert_close(reloaded.state_dict()[key], tensor, rtol=0, atol=0)
    state = load_file(native_dir / "model.safetensors")
    state.pop(next(key for key in state if key.endswith(".lora_b")))
    save_file(state, native_dir / "model.safetensors")
    with pytest.raises(RuntimeError, match="Missing key"):
        tiny_policy.from_pretrained(native_dir, local_files_only=True)
    state = {key: value for key, value in state.items() if not key.endswith((".lora_a", ".lora_b"))}
    save_file(state, native_dir / "model.safetensors")
    with pytest.raises(RuntimeError, match="Missing key"):
        tiny_policy.from_pretrained(native_dir, local_files_only=True)


def test_native_lora_loads_a_weights_only_base(tiny_policy, tmp_path):
    base = tiny_policy(PI05Config(device="cpu", push_to_hub=False))
    save_file(base.state_dict(), tmp_path / "model.safetensors")
    config = PI05Config(device="cpu", native_lora=PI05NativeLoRAConfig(rank=4, alpha=4))
    native = tiny_policy.from_pretrained(tmp_path, config=config, local_files_only=True)
    for key, value in base.state_dict().items():
        torch.testing.assert_close(native.state_dict()[key], value, rtol=0, atol=0)


@pytest.mark.parametrize("native_lora", [None, PI05NativeLoRAConfig()])
def test_missing_weights_do_not_return_random_policy(tiny_policy, tmp_path, native_lora):
    config = PI05Config(device="cpu", native_lora=native_lora)
    with pytest.raises(OSError):
        tiny_policy.from_pretrained(tmp_path, config=config, local_files_only=True)


def test_native_lora_cannot_be_combined_with_peft():
    from lerobot.configs.default import PeftConfig

    policy = PI05Config(native_lora=PI05NativeLoRAConfig(), push_to_hub=False)
    cfg = TrainPipelineConfig(dataset=SimpleNamespace(), policy=policy, peft=PeftConfig())
    with pytest.raises(ValueError, match="cannot be enabled together"):
        cfg.validate()
    with pytest.raises(ValueError, match="cannot be combined"):
        PI05Config(native_lora=PI05NativeLoRAConfig(), use_peft=True)


@pytest.mark.parametrize("use_memory", [False, True])
def test_openpi_precision_runs_siglip_with_fp32_embeddings_and_bf16_encoder(use_memory):
    from transformers.models.siglip.configuration_siglip import SiglipVisionConfig
    from transformers.models.siglip.modeling_siglip import SiglipVisionTransformer

    vision = SiglipVisionTransformer(
        SiglipVisionConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            image_size=16,
            patch_size=4,
            vision_use_head=False,
        )
    )
    wrapper = PaliGemmaWithExpertModel.__new__(PaliGemmaWithExpertModel)
    nn.Module.__init__(wrapper)
    wrapper.use_openpi_precision = True
    wrapper.paligemma = nn.Module()
    wrapper.paligemma.model = nn.Module()
    wrapper.paligemma.model.vision_tower = nn.Module()
    wrapper.paligemma.model.vision_tower.vision_model = vision
    wrapper.paligemma.model.multi_modal_projector = nn.Linear(16, 8)
    wrapper.to_bfloat16_for_selected_params("bfloat16")
    assert vision.embeddings.patch_embedding.weight.dtype == torch.float32
    assert vision.embeddings.position_embedding.weight.dtype == torch.float32
    assert vision.encoder.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16
    assert wrapper.paligemma.model.multi_modal_projector.weight.dtype == torch.bfloat16
    images = torch.randn(2, 3, 16, 16)
    if use_memory:
        images = images[:, None]
    out = wrapper.embed_image(images, temporal_attention_every=1)
    assert out.shape == (2, 16, 8)
    out.square().mean().backward()
    assert torch.isfinite(vision.encoder.layers[0].self_attn.q_proj.weight.grad).all()
    wrapper.use_openpi_precision = False
    wrapper.to_bfloat16_for_selected_params("bfloat16")
    assert all(p.dtype == torch.float32 for p in vision.parameters())
