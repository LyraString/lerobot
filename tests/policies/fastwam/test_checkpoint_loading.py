# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

import pytest
import safetensors.torch
import torch
from torch import nn

pytest.importorskip("transformers", reason="fastwam requires the `fastwam` extra (transformers)")
pytest.importorskip("diffusers", reason="fastwam requires the `fastwam` extra (diffusers)")

from lerobot.policies.fastwam.configuration_fastwam import FastWAMConfig
from lerobot.policies.fastwam.modeling_fastwam import FastWAMPolicy


@pytest.fixture
def cpu_checkpoint_reads(monkeypatch):
    """Reject a full checkpoint allocation on CUDA, even on CPU-only test hosts."""
    original = safetensors.torch.load_file
    reads = []

    def load_file(filename, device="cpu", **kwargs):
        assert torch.device(device).type == "cpu", "Checkpoint tensors must be staged on CPU"
        reads.append(filename)
        return original(filename, device=device, **kwargs)

    monkeypatch.setattr(safetensors.torch, "load_file", load_file)
    return reads


@pytest.mark.parametrize("strict", [False, True])
def test_fp32_checkpoint_loads_into_bf16_parameters_without_cuda_staging(
    tmp_path, cpu_checkpoint_reads, strict
):
    source = nn.Linear(2, 3)
    with torch.no_grad():
        source.weight.fill_(0.123456)
        source.bias.fill_(-0.234567)
    checkpoint = tmp_path / "model.safetensors"
    safetensors.torch.save_model(source, checkpoint)
    target = nn.Linear(2, 3, dtype=torch.bfloat16)

    result = FastWAMPolicy._load_as_safetensor(target, str(checkpoint), "cuda:0", strict)

    assert result is target
    assert len(cpu_checkpoint_reads) == 1
    for name, value in target.state_dict().items():
        assert value.dtype == torch.bfloat16
        torch.testing.assert_close(value, source.state_dict()[name].to(torch.bfloat16), rtol=0, atol=0)


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("missing_bias", [False, True])
def test_cpu_staging_preserves_missing_and_unexpected_key_handling(
    tmp_path, cpu_checkpoint_reads, strict, missing_bias
):
    source = nn.Linear(2, 3, bias=not missing_bias)
    target = nn.Linear(2, 3, bias=missing_bias)
    checkpoint = tmp_path / "model.safetensors"
    safetensors.torch.save_model(source, checkpoint)

    if strict:
        with pytest.raises(RuntimeError, match="bias"):
            FastWAMPolicy._load_as_safetensor(target, str(checkpoint), "cuda:0", strict=True)
    else:
        FastWAMPolicy._load_as_safetensor(target, str(checkpoint), "cuda:0", strict=False)
        torch.testing.assert_close(target.weight, source.weight, rtol=0, atol=0)


@pytest.mark.parametrize("strict", [False, True])
def test_cross_embodiment_loading_preserves_compatible_weights(tmp_path, cpu_checkpoint_reads, strict):
    source = nn.ModuleDict({"head": nn.Linear(2, 3), "shared": nn.Linear(2, 2)})
    target = nn.ModuleDict({"head": nn.Linear(2, 4), "shared": nn.Linear(2, 2)})
    original = {name: value.clone() for name, value in target.state_dict().items()}
    checkpoint = tmp_path / "model.safetensors"
    safetensors.torch.save_model(source, checkpoint)

    if strict:
        with pytest.raises(RuntimeError, match="shape mismatch"):
            FastWAMPolicy._load_as_safetensor(target, str(checkpoint), "cpu", strict=True)
        for name, value in target.state_dict().items():
            torch.testing.assert_close(value, original[name], rtol=0, atol=0)
    else:
        FastWAMPolicy._load_as_safetensor(target, str(checkpoint), "cpu", strict=False)
        for name, value in target.state_dict().items():
            expected = original[name] if name.startswith("head.") else source.state_dict()[name]
            torch.testing.assert_close(value, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda:0", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA"))],
)
def test_from_pretrained_keeps_requested_device_and_dtype_after_cpu_staging(
    monkeypatch, tmp_path, cpu_checkpoint_reads, device
):
    monkeypatch.setattr(
        FastWAMPolicy,
        "_build_core_model",
        lambda self, config: nn.Linear(2, 2).to(device=config.device, dtype=torch.bfloat16),
    )
    config = FastWAMConfig(device="cpu", torch_dtype="bfloat16", base_model_id=None)
    reference = FastWAMPolicy(config).float()
    with torch.no_grad():
        reference.model.weight.fill_(0.123456)
        reference.model.bias.fill_(-0.234567)
    reference.save_pretrained(tmp_path)
    config.device = device

    loaded = FastWAMPolicy.from_pretrained(tmp_path, config=config, strict=True)

    assert len(cpu_checkpoint_reads) == 1
    for name, value in loaded.state_dict().items():
        assert value.device == torch.device(device)
        assert value.dtype == torch.bfloat16
        expected = reference.state_dict()[name].to(device=device, dtype=torch.bfloat16)
        torch.testing.assert_close(value, expected, rtol=0, atol=0)
