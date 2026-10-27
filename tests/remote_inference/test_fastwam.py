"""FastWAM's real adapter/processors over remote inference, with a tiny model core."""

import time
from dataclasses import replace
from threading import Event, Thread

import numpy as np
import pytest
import torch
from torch import nn

pytest.importorskip("transformers")
pytest.importorskip("diffusers")

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.inference import ExecutionMode, FeatureSpec, ObservationSnapshot, RemoteInferenceConfig
from lerobot.policies import make_pre_post_processors
from lerobot.policies.fastwam.configuration_fastwam import FastWAMConfig
from lerobot.policies.fastwam.modeling_fastwam import FastWAMPolicy
from lerobot.policies.fastwam.processor_fastwam import (
    FastWAMTextContextProcessorStep,
    text_context_config,
)
from lerobot.remote_inference import PolicyServer, SessionWorker, load_deployment
from lerobot.remote_inference.client import RemoteClient
from lerobot.remote_inference.configs import ExecutionConfig, ModelConfig, ServerConfig
from lerobot.transport import ZenohConfig, ZenohTransport
from tests.remote_inference.helpers import admit
from tests.remote_inference.network_helpers import free_port, runtime_for, signal_server_ready

CAMERAS = ("observation.images.front", "observation.images.wrist")
NAMES = ("a.pos", "b.pos", "gripper.pos")


class TinyCore(nn.Module):
    """Replace GPU weights, while leaving FastWAM batch adaptation in use."""

    def __init__(self):
        super().__init__()
        self.dit = nn.Linear(3, 3, bias=False)
        with torch.no_grad():
            self.dit.weight.copy_(torch.eye(3))

    def infer_action(self, **kwargs):
        assert kwargs["input_image"].shape == (1, 3, 2, 4)
        values = self.dit(kwargs["proprio"]) + kwargs["input_image"].mean()
        if kwargs.get("context") is not None:
            assert kwargs["prompt"] is None, "The core forbids simultaneous prompt and cached context"
            values = values + kwargs["context"].mean()
        return {"action": values.unsqueeze(1).repeat(1, kwargs["action_horizon"], 1)}


@pytest.fixture
def saved_fastwam(tmp_path, monkeypatch):
    monkeypatch.setattr(FastWAMPolicy, "_build_core_model", lambda *_: TinyCore())
    config = FastWAMConfig(
        device="cpu",
        torch_dtype="float32",
        base_model_id=None,
        action_dim=3,
        proprio_dim=3,
        action_horizon=4,
        n_action_steps=2,
        num_video_frames=5,
        action_video_freq_ratio=1,
        image_size=(2, 4),
        input_features={
            **{name: PolicyFeature(FeatureType.VISUAL, (3, 2, 2)) for name in CAMERAS},
            "observation.state": PolicyFeature(FeatureType.STATE, (3,)),
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (3,))},
    )
    policy = FastWAMPolicy(config)
    pre, post = make_pre_post_processors(
        config,
        dataset_stats={
            "observation.state": {"mean": torch.ones(3), "std": torch.ones(3) * 2},
            "action": {"mean": torch.tensor([1.0, 2.0, 3.0]), "std": torch.ones(3) * 3},
        },
    )
    policy.save_pretrained(tmp_path)
    pre.save_pretrained(tmp_path)
    post.save_pretrained(tmp_path)
    return ServerConfig(
        deployment="fastwam-test",
        semantics="radians-gripper-ratio-v1",
        model=ModelConfig(str(tmp_path), device="cpu"),
        execution=ExecutionConfig(action_fps=20, warmup_calls=1),
        zenoh=ZenohConfig(listen_endpoints=["tcp/127.0.0.1:7447"]),
        features=[
            FeatureSpec("observation.state", (3,), "float32", names=NAMES, semantics="radians"),
            *(FeatureSpec(name, (6, 8, 3), "uint8", kind="rgb", semantics=name) for name in CAMERAS),
        ],
        action_feature=FeatureSpec("action", (3,), "float32", names=NAMES, semantics="radians"),
    )


def snapshot():
    return ObservationSnapshot(
        features={
            "observation.state": np.array([2.0, 3.0, 0.5], dtype=np.float32),
            **{name: np.full((6, 8, 3), 128, dtype=np.uint8) for name in CAMERAS},
        },
        capture_time=time.monotonic(),
        task="pick up the object",
        task_version=0,
        observation_id="fastwam-observation",
    )


def test_saved_fastwam_runner_matches_select_action_and_processor_pipeline(saved_fastwam):
    runner, identity = load_deployment(saved_fastwam)
    assert identity.startswith("sha256:")
    assert runner.capabilities.modes == (ExecutionMode.CHUNK,)
    assert runner.capabilities.prediction_steps == 4
    assert runner.capabilities.execution_steps == 2
    observation = snapshot()
    batch = runner.preprocessor(runner._batch(observation))
    with torch.inference_mode():
        expected = torch.stack(
            [runner.postprocessor(runner.policy.select_action(batch))[0] for _ in range(2)]
        )
    runner.reset()
    torch.testing.assert_close(runner.predict(observation).canonical_actions, expected)
    assert not runner.policy._action_queue


def test_fastwam_rejects_bad_camera_state_and_rtc_contracts(saved_fastwam):
    runner, _ = load_deployment(saved_fastwam)
    features = runner.capabilities.features
    with pytest.raises(ValueError, match="configured inputs"):
        runner.policy.validate_chunk_input_features(features[:-1])
    with pytest.raises(ValueError, match="Tensor feature shape"):
        runner.policy.validate_chunk_input_features(
            (replace(features[0], shape=(4,), names=()), *features[1:])
        )
    with pytest.raises(ValueError, match="RGB camera"):
        runner.policy.validate_chunk_input_features((*features[:-1], replace(features[-1], kind="tensor")))
    with pytest.raises(ValueError, match="not enabled"):
        runner.predict(snapshot(), mode=ExecutionMode.RTC_GUIDED)
    runner.policy.config.n_obs_steps = 2
    with pytest.raises(ValueError, match="n_obs_steps=1"):
        runner.policy.chunk_inference_spec()


def test_saved_text_cache_supports_task_warmup_without_original_training_cache(saved_fastwam):
    path = saved_fastwam.model.repo_or_path
    config = FastWAMConfig.from_pretrained(path)
    pre, _ = make_pre_post_processors(config, pretrained_path=path)
    config.load_text_encoder = False
    config.text_context_cache = "/absent/training-only/text-cache.safetensors"
    config.optimizer_fused = True
    config.save_pretrained(path)
    cached = FastWAMTextContextProcessorStep(
        tasks=[snapshot().task], encoding_config=text_context_config(config)
    )
    cached.load_state_dict({"context": torch.ones(1, 2, 4), "context_mask": torch.ones(1, 2)})
    pre.steps.insert(0, cached)
    pre.save_pretrained(path)
    deployment = replace(
        saved_fastwam,
        execution=replace(saved_fastwam.execution, warmup_task=snapshot().task),
    )

    runner, _ = load_deployment(deployment)
    assert runner.policy.config.optimizer_fused is True
    assert runner.policy.config.get_optimizer_preset().fused is True
    batch = runner.preprocessor(runner._batch(snapshot()))
    torch.testing.assert_close(batch["context"], torch.ones(1, 2, 4))
    with torch.inference_mode():
        expected = runner.postprocessor(runner.policy.select_action(batch))[0]
    runner.reset()
    result = runner.predict(snapshot()).canonical_actions
    torch.testing.assert_close(result, expected.expand(2, -1))
    with pytest.raises(KeyError):
        runner.predict(replace(snapshot(), task="task absent from the saved cache"))


@pytest.mark.parametrize("encoding", ["raw", "jpeg"])
def test_fastwam_saved_processors_roundtrip_through_zenoh(saved_fastwam, encoding):
    pytest.importorskip("zenoh")
    runner, identity = load_deployment(saved_fastwam)
    observation = snapshot()
    expected = runner.predict(observation).canonical_actions
    worker = SessionWorker(
        runner, deployment="fastwam-test", artifact_identity=identity, semantics=saved_fastwam.semantics
    )
    endpoint = f"tcp/127.0.0.1:{free_port()}"
    transport = ZenohTransport(ZenohConfig(listen_endpoints=[endpoint]))
    server = PolicyServer(worker, transport)
    ready = Event()
    signal_server_ready(transport, ready)
    thread = Thread(target=server.serve, daemon=True)
    client = None
    try:
        thread.start()
        assert ready.wait(10)
        client = RemoteClient.connect(
            RemoteInferenceConfig(
                endpoint=endpoint,
                deployment="fastwam-test",
                semantics=saved_fastwam.semantics,
                encoding=encoding,
                chunk_merge="append",
            )
        )
        admit(client)
        runtime = runtime_for(client)
        request = runtime.begin(replace(observation, capture_time=time.monotonic()))
        result = client.infer(request)
        torch.testing.assert_close(result.canonical_actions, expected)
        assert result.provenance.session_id == client.session_id
    finally:
        if client is not None:
            client.close()
        server.stop()
        thread.join(10)
        transport.close()
        worker.close()
        assert not thread.is_alive()
