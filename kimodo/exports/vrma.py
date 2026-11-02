# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Export Kimodo SOMA motions as VRM Animation (``.vrma``) files.

A ``.vrma`` is a glTF binary (GLB) carrying the ``VRMC_vrm_animation``
extension: named humanoid bone nodes whose ``rotation`` channels hold local
bone rotations, plus an optional ``translation`` channel on the ``hips`` node.

The nodes carry no rest rotation, so a loader's rest-space conversion collapses
to identity and the stored tracks are exactly the local bone quaternions.
Kimodo's ``joints_local_rot`` are expressed in a Y-up, meter-scale, standard
T-pose frame which matches the VRM humanoid frame, so no axis flip or mirroring
is required.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Union

import numpy as np
import torch

from kimodo.geometry import matrix_to_quaternion

# Ordered mapping of VRM 1.0 humanoid bone names to the SOMA joint chains that
# drive them. Most bones map 1:1; the VRM ``neck`` bone collapses SOMA's
# ``Neck1``/``Neck2`` chain, whose local rotations are composed.
VRM_BONE_CHAINS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("hips", ("Hips",)),
    ("spine", ("Spine1",)),
    ("chest", ("Spine2",)),
    ("upperChest", ("Chest",)),
    ("neck", ("Neck1", "Neck2")),
    ("head", ("Head",)),
    ("leftShoulder", ("LeftShoulder",)),
    ("leftUpperArm", ("LeftArm",)),
    ("leftLowerArm", ("LeftForeArm",)),
    ("leftHand", ("LeftHand",)),
    ("rightShoulder", ("RightShoulder",)),
    ("rightUpperArm", ("RightArm",)),
    ("rightLowerArm", ("RightForeArm",)),
    ("rightHand", ("RightHand",)),
    ("leftUpperLeg", ("LeftLeg",)),
    ("leftLowerLeg", ("LeftShin",)),
    ("leftFoot", ("LeftFoot",)),
    ("leftToes", ("LeftToeBase",)),
    ("rightUpperLeg", ("RightLeg",)),
    ("rightLowerLeg", ("RightShin",)),
    ("rightFoot", ("RightFoot",)),
    ("rightToes", ("RightToeBase",)),
)


def _coerce_clip(name: str, x: torch.Tensor, *, expected_ndim: int) -> torch.Tensor:
    """Coerce ``(T, ...)`` or ``(1, T, ...)`` into ``(T, ...)``."""
    if x.ndim == expected_ndim:
        return x
    if x.ndim == expected_ndim + 1:
        if int(x.shape[0]) != 1:
            raise ValueError(
                f"{name} has batch dimension B={int(x.shape[0])}, but VRMA export "
                "only supports a single clip (B==1)."
            )
        return x[0]
    raise ValueError(f"{name} must have shape (T, ...) or (1, T, ...); got {tuple(x.shape)}")


def _align4(buffer: bytearray) -> None:
    while len(buffer) % 4:
        buffer.append(0)


def _append_view(buffer: bytearray, raw: bytes) -> tuple[int, int]:
    """Append ``raw`` to ``buffer`` at a 4-byte aligned offset; return (offset, length)."""
    _align4(buffer)
    offset = len(buffer)
    buffer.extend(raw)
    return offset, len(raw)


def _soma77_local_rots(local_rot_mats: torch.Tensor, skeleton) -> torch.Tensor:
    """Normalize local rotations to the 77-joint SOMA layout as a CPU float tensor."""
    if local_rot_mats.ndim != 4 or local_rot_mats.shape[-2:] != (3, 3):
        raise ValueError(f"local_rot_mats must have shape (T, J, 3, 3); got {tuple(local_rot_mats.shape)}")

    n_joints = int(local_rot_mats.shape[1])
    if n_joints == 30:
        if skeleton is None or not hasattr(skeleton, "to_SOMASkeleton77"):
            raise ValueError("A SOMA skeleton is required to expand 30-joint rotations to 77 joints.")
        local_rot_mats = skeleton.to_SOMASkeleton77(local_rot_mats)
    elif n_joints != 77:
        raise ValueError(
            f"VRMA export expects 30- or 77-joint SOMA rotations; got J={n_joints}. "
            "VRMA is only supported for SOMA models."
        )

    return local_rot_mats.detach().to(device="cpu", dtype=torch.float32)


def _quats_xyzw(mats: torch.Tensor) -> np.ndarray:
    """Convert ``(T, 3, 3)`` rotation matrices to ``(T, 4)`` xyzw quaternions.

    Quaternion signs are made continuous over time so linear interpolation does
    not take the long way around between frames.
    """
    quats = matrix_to_quaternion(mats)  # (T, 4), wxyz
    quats = quats / torch.clamp(quats.norm(dim=-1, keepdim=True), min=1e-8)
    quats = quats.numpy()
    for t in range(1, len(quats)):
        if float(np.dot(quats[t], quats[t - 1])) < 0.0:
            quats[t] = -quats[t]
    # wxyz -> xyzw
    return quats[:, [1, 2, 3, 0]].astype(np.float32)


def _rest_hips_height(skeleton) -> float | None:
    """Hip height above the lowest joint in the skeleton's neutral pose, if available."""
    # Rotations are always emitted in the 77-joint layout, so measure the rest
    # pose of that skeleton when a compact (30-joint) one was supplied.
    skeleton = getattr(skeleton, "somaskel77", skeleton)
    neutral = getattr(skeleton, "neutral_joints", None)
    if neutral is None:
        return None
    try:
        neutral = neutral.detach().to(device="cpu", dtype=torch.float64).numpy()
        root_idx = int(getattr(skeleton, "root_idx", 0) or 0)
        height = float(neutral[root_idx, 1] - neutral[:, 1].min())
    except Exception:
        return None
    if np.isfinite(height) and height >= 0.1:
        return height
    return None


def motion_to_vrma_bytes(
    local_rot_mats: torch.Tensor,
    root_positions: torch.Tensor,
    *,
    skeleton,
    fps: float,
) -> bytes:
    """Convert local rotations and root positions to VRM Animation (``.vrma``) bytes.

    Args:
        local_rot_mats: ``(T, J, 3, 3)`` or ``(1, T, J, 3, 3)`` local rotation
            matrices, with ``J`` either 30 or 77 (SOMA skeletons).
        root_positions: ``(T, 3)`` or ``(1, T, 3)`` absolute root joint positions.
        skeleton: SOMA skeleton used to expand 30-joint rotations to 77 joints.
        fps: Frames per second for the motion.

    Returns:
        GLB bytes containing a ``VRMC_vrm_animation`` 1.0 animation.
    """
    local_rot_mats = _coerce_clip("local_rot_mats", local_rot_mats, expected_ndim=4)
    root_positions = _coerce_clip("root_positions", root_positions, expected_ndim=2)

    if root_positions.shape[-1] != 3:
        raise ValueError(f"root_positions must have shape (T, 3); got {tuple(root_positions.shape)}")
    if local_rot_mats.shape[0] != root_positions.shape[0]:
        raise ValueError(
            "local_rot_mats and root_positions must have the same number of frames; "
            f"got {int(local_rot_mats.shape[0])} and {int(root_positions.shape[0])}."
        )
    fps = float(fps)
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"fps must be a positive finite number; got {fps}")

    local_rot_mats = _soma77_local_rots(local_rot_mats, skeleton)
    root_positions = root_positions.detach().to(device="cpu", dtype=torch.float32)

    soma_names = list(getattr(skeleton, "bone_order_names", []) or [])
    if len(soma_names) != 77 and hasattr(skeleton, "somaskel77"):
        soma_names = list(skeleton.somaskel77.bone_order_names)
    if len(soma_names) != 77:
        # Fall back to the canonical 77-joint skeleton for name resolution.
        from kimodo.skeleton.definitions import SOMASkeleton77

        soma_names = SOMASkeleton77().bone_order_names
    name_to_index = {name: i for i, name in enumerate(soma_names)}

    # A single frame cannot be animated; repeat it so samplers have two keys.
    if int(local_rot_mats.shape[0]) == 1:
        local_rot_mats = local_rot_mats.repeat(2, 1, 1, 1)
        root_positions = root_positions.repeat(2, 1)

    n_frames = int(local_rot_mats.shape[0])
    times = (np.arange(n_frames, dtype=np.float32) / fps).astype(np.float32)
    if n_frames > 1 and not np.all(np.diff(times) > 0):
        raise ValueError("frame times must be strictly increasing; check the fps value.")

    # Rest hips height used by loaders to scale the translation track:
    # scale = avatar rest hips height / this value. Prefer the skeleton's
    # neutral hip height (stable across crouches/jumps); fall back to the
    # clip's first frame and finally to 1 m.
    hips_rest_y = _rest_hips_height(skeleton)
    if hips_rest_y is None:
        hips_rest_y = float(root_positions[0, 1])
    if not np.isfinite(hips_rest_y) or hips_rest_y < 0.1:
        hips_rest_y = float(root_positions[:, 1].max())
    if not np.isfinite(hips_rest_y) or hips_rest_y < 0.1:
        hips_rest_y = 1.0

    # Resolve bone chains into composed local rotations.
    bone_rotations: list[tuple[str, torch.Tensor]] = []
    for vrm_name, chain in VRM_BONE_CHAINS:
        missing = [name for name in chain if name not in name_to_index]
        if missing:
            raise ValueError(f"Skeleton is missing joint(s) {missing} required for VRM bone {vrm_name!r}.")
        mats = local_rot_mats[:, name_to_index[chain[0]]]
        for name in chain[1:]:
            mats = mats @ local_rot_mats[:, name_to_index[name]]
        bone_rotations.append((vrm_name, mats))

    nodes: list[dict] = []
    human_bones: dict[str, dict] = {}
    rotation_channels: list[tuple[int, np.ndarray]] = []
    for vrm_name, mats in bone_rotations:
        node = {"name": vrm_name}
        if vrm_name == "hips":
            # Rest hips height; loaders scale the translation track by
            # (avatar rest hips height / this value).
            node["translation"] = [0.0, hips_rest_y, 0.0]
        human_bones[vrm_name] = {"node": len(nodes)}
        nodes.append(node)
        rotation_channels.append((len(nodes) - 1, _quats_xyzw(mats)))

    hips_node = human_bones["hips"]["node"]
    hips_translation = root_positions.numpy().astype(np.float32)

    buffer = bytearray()
    buffer_views: list[dict] = []
    accessors: list[dict] = []
    samplers: list[dict] = []
    channels: list[dict] = []

    # Shared time accessor.
    time_raw = times.tobytes()
    buffer.extend(time_raw)
    buffer_views.append({"buffer": 0, "byteOffset": 0, "byteLength": len(time_raw)})
    accessors.append(
        {
            "bufferView": len(buffer_views) - 1,
            "componentType": 5126,
            "count": n_frames,
            "type": "SCALAR",
            "min": [float(times[0])],
            "max": [float(times[-1])],
        }
    )
    time_accessor = len(accessors) - 1

    def add_channel(node_index: int, path: str, values: np.ndarray) -> None:
        raw = np.asarray(values, dtype=np.float32).tobytes()
        offset, length = _append_view(buffer, raw)
        buffer_views.append({"buffer": 0, "byteOffset": offset, "byteLength": length})
        accessors.append(
            {
                "bufferView": len(buffer_views) - 1,
                "componentType": 5126,
                "count": int(values.shape[0]),
                "type": "VEC4" if values.shape[-1] == 4 else "VEC3",
            }
        )
        samplers.append(
            {"input": time_accessor, "output": len(accessors) - 1, "interpolation": "LINEAR"}
        )
        channels.append({"sampler": len(samplers) - 1, "target": {"node": node_index, "path": path}})

    for node_index, quats in rotation_channels:
        add_channel(node_index, "rotation", quats)
    add_channel(hips_node, "translation", hips_translation)

    gltf = {
        "asset": {"version": "2.0", "generator": "kimodo"},
        "scene": 0,
        "scenes": [{"nodes": list(range(len(nodes)))}],
        "nodes": nodes,
        "animations": [{"channels": channels, "samplers": samplers}],
        "bufferViews": buffer_views,
        "buffers": [{"byteLength": len(buffer)}],
        "accessors": accessors,
        "extensionsUsed": ["VRMC_vrm_animation"],
        "extensions": {
            "VRMC_vrm_animation": {
                "specVersion": "1.0",
                "humanoid": {"humanBones": human_bones},
            }
        },
    }

    _align4(buffer)
    json_chunk = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    while len(json_chunk) % 4:
        json_chunk += b" "

    total = 12 + 8 + len(json_chunk) + 8 + len(buffer)
    blob = bytearray()
    blob += struct.pack("<III", 0x46546C67, 2, total)
    blob += struct.pack("<II", len(json_chunk), 0x4E4F534A) + json_chunk
    blob += struct.pack("<II", len(buffer), 0x004E4942) + buffer
    return bytes(blob)


def save_motion_vrma(
    path: Union[str, Path],
    local_rot_mats: torch.Tensor,
    root_positions: torch.Tensor,
    *,
    skeleton,
    fps: float,
) -> None:
    """Write local rotations and root positions to a ``.vrma`` file at ``path``."""
    data = motion_to_vrma_bytes(local_rot_mats, root_positions, skeleton=skeleton, fps=fps)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
