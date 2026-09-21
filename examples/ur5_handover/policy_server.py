#!/usr/bin/env python

"""Serve SmolVLA action chunks to the ManiSkill UR5 handover evaluator."""

from __future__ import annotations

import argparse
from contextlib import closing
from io import BytesIO
from pathlib import Path
import socket
import struct

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


CAMERA_NAMES = ("head_camera", "left_wrist_camera", "right_wrist_camera")
DEFAULT_CHECKPOINT = Path(
    "outputs/train/ur5_handover_smolvla_full_50k/checkpoints/050000/pretrained_model"
)
MAX_PACKET_BYTES = 64 * 1024 * 1024
PACKET_LENGTH = struct.Struct("!Q")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Load the checkpoint and processors, then exit without listening.",
    )
    return parser.parse_args()


def _recv_exact(connection: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = connection.recv(size - len(chunks))
        if not chunk:
            raise EOFError("Connection closed while receiving a packet.")
        chunks.extend(chunk)
    return bytes(chunks)


def recv_packet(connection: socket.socket) -> dict[str, np.ndarray]:
    size = PACKET_LENGTH.unpack(_recv_exact(connection, PACKET_LENGTH.size))[0]
    if size <= 0 or size > MAX_PACKET_BYTES:
        raise ValueError(f"Invalid packet size: {size} bytes")
    payload = _recv_exact(connection, size)
    with np.load(BytesIO(payload), allow_pickle=False) as packet:
        return {key: packet[key] for key in packet.files}


def send_packet(connection: socket.socket, **arrays: np.ndarray) -> None:
    stream = BytesIO()
    np.savez(stream, **arrays)
    payload = stream.getvalue()
    if len(payload) > MAX_PACKET_BYTES:
        raise ValueError(f"Packet exceeds {MAX_PACKET_BYTES} bytes")
    connection.sendall(PACKET_LENGTH.pack(len(payload)) + payload)


def _scalar_string(packet: dict[str, np.ndarray], key: str) -> str:
    if key not in packet:
        raise KeyError(f"Missing packet field: {key}")
    value = np.asarray(packet[key])
    if value.shape != () or value.dtype.kind not in ("U", "S"):
        raise ValueError(f"{key} must be a scalar string, got shape={value.shape}, dtype={value.dtype}")
    return str(value.item())


class SmolVLAServer:
    def __init__(self, checkpoint: Path, device: str):
        checkpoint = checkpoint.expanduser().resolve()
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint}")

        self.config = PreTrainedConfig.from_pretrained(checkpoint)
        if self.config.type != "smolvla":
            raise ValueError(f"Expected a SmolVLA checkpoint, found {self.config.type!r}")
        self.config.device = device
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; pass --device=cpu or fix the CUDA environment.")

        self.policy = SmolVLAPolicy.from_pretrained(
            checkpoint,
            config=self.config,
            strict=True,
        ).to(device)
        self.policy.eval()
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.config,
            pretrained_path=str(checkpoint),
            preprocessor_overrides={"device_processor": {"device": device}},
        )

        self.expected_state_shape = tuple(self.config.input_features["observation.state"].shape)
        self.expected_action_shape = (
            int(self.config.chunk_size),
            *tuple(self.config.action_feature.shape),
        )
        self.expected_images = {
            name: tuple(self.config.input_features[f"observation.images.{name}"].shape)
            for name in CAMERA_NAMES
        }
        self.reset()

    def reset(self) -> None:
        self.policy.reset()

    def predict(self, packet: dict[str, np.ndarray]) -> np.ndarray:
        state = np.asarray(packet.get("state"), dtype=np.float32)
        if state.shape != self.expected_state_shape:
            raise ValueError(f"Expected state shape {self.expected_state_shape}, got {state.shape}")
        if not np.isfinite(state).all():
            raise ValueError("State contains NaN or Inf.")

        task = _scalar_string(packet, "task")
        batch: dict[str, torch.Tensor | str] = {
            "observation.state": torch.from_numpy(np.ascontiguousarray(state)),
            "task": task,
        }
        for name, chw_shape in self.expected_images.items():
            if name not in packet:
                raise KeyError(f"Missing packet field: {name}")
            image = np.asarray(packet[name])
            hwc_shape = (chw_shape[1], chw_shape[2], chw_shape[0])
            if image.dtype != np.uint8 or image.shape != hwc_shape:
                raise ValueError(
                    f"Expected {name} uint8 HWC shape {hwc_shape}, "
                    f"got shape={image.shape}, dtype={image.dtype}"
                )
            image_chw = np.ascontiguousarray(image.transpose(2, 0, 1))
            batch[f"observation.images.{name}"] = torch.from_numpy(image_chw).float().div_(255.0)

        seed = int(np.asarray(packet.get("seed", np.int64(0))).reshape(()).item())
        torch.manual_seed(seed)
        processed = self.preprocessor(batch)
        with torch.inference_mode():
            normalized = self.policy.predict_action_chunk(processed)
            actions = self.postprocessor(normalized)
        actions = actions.detach().cpu().numpy()
        expected = (1, *self.expected_action_shape)
        if actions.shape != expected:
            raise RuntimeError(f"Expected action chunk shape {expected}, got {actions.shape}")
        if not np.isfinite(actions).all():
            raise RuntimeError("Policy returned NaN or Inf actions.")
        return np.asarray(actions[0], dtype=np.float32)


def serve_client(connection: socket.socket, server: SmolVLAServer) -> None:
    while True:
        try:
            packet = recv_packet(connection)
        except EOFError:
            return

        try:
            command = _scalar_string(packet, "command")
            if command == "reset":
                server.reset()
                send_packet(connection, status=np.asarray("ok"))
            elif command == "predict":
                actions = server.predict(packet)
                send_packet(connection, status=np.asarray("ok"), actions=actions)
            else:
                raise ValueError(f"Unknown command: {command!r}")
        except Exception as error:
            send_packet(
                connection,
                status=np.asarray("error"),
                error=np.asarray(f"{type(error).__name__}: {error}"),
            )


def main() -> None:
    args = parse_args()
    server = SmolVLAServer(args.checkpoint, args.device)
    print(
        f"Loaded {args.checkpoint} on {args.device}; output shape={server.expected_action_shape}",
        flush=True,
    )
    if args.check_only:
        return

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((args.host, args.port))
        listener.listen(1)
        print(f"Listening on {args.host}:{args.port}", flush=True)
        while True:
            connection, address = listener.accept()
            print(f"Client connected: {address[0]}:{address[1]}", flush=True)
            with connection:
                serve_client(connection, server)
            print("Client disconnected", flush=True)


if __name__ == "__main__":
    main()
