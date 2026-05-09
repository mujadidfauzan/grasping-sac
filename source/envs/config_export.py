from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def to_config_value(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): to_config_value(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_config_value(item) for item in value]
    return value


def capture_init_config(init_locals: dict) -> dict:
    config = {}
    for key, value in init_locals.items():
        if key in {"self", "__class__", "kwargs"}:
            continue
        config[key] = to_config_value(value)

    extra_kwargs = init_locals.get("kwargs", {})
    if extra_kwargs:
        config["extra_kwargs"] = to_config_value(extra_kwargs)

    return config


def describe_space(space) -> dict:
    return {
        "type": type(space).__name__,
        "shape": list(space.shape),
        "dtype": str(space.dtype),
        "low": np.asarray(space.low, dtype=np.float64).tolist(),
        "high": np.asarray(space.high, dtype=np.float64).tolist(),
    }


def build_observation_layout(obs_components: list[tuple[str, np.ndarray]]) -> list[dict]:
    layout = []
    start = 0
    for name, component in obs_components:
        flat_component = np.asarray(component, dtype=np.float64).reshape(-1)
        size = int(flat_component.size)
        layout.append(
            {
                "name": name,
                "size": size,
                "start": start,
                "end": start + size,
            }
        )
        start += size

    return layout


def group_init_config(init_config: dict) -> dict[str, dict]:
    reward = {}
    action = {}
    simulation = {}
    task = {}

    for key, value in init_config.items():
        if key.startswith("reward_") or key.endswith("_penalty_weight"):
            reward[key] = value
        elif "action" in key:
            action[key] = value
        elif key in {"xml_file", "frame_skip", "default_camera_config"}:
            simulation[key] = value
        else:
            task[key] = value

    return {
        "reward": reward,
        "action": action,
        "simulation": simulation,
        "task": task,
    }


def export_env_config(env, obs_components: list[tuple[str, np.ndarray]]) -> dict:
    grouped_config = group_init_config(env._init_config)
    return {
        "env_name": type(env).__name__,
        "xml_file": str(env.fullpath),
        "init": dict(env._init_config),
        "reward": {
            "params": grouped_config["reward"],
        },
        "action": {
            "params": grouped_config["action"],
            "space": describe_space(env.action_space),
            "arm_ctrl_dim": int(env._arm_ctrl_dim),
            "ctrl_low": np.asarray(env._ctrl_low, dtype=np.float64).tolist(),
            "ctrl_high": np.asarray(env._ctrl_high, dtype=np.float64).tolist(),
            "gripper_policy": "heuristic",
            "gripper_open_target": [0.01, -0.01],
            "gripper_closed_target": [-0.02, 0.02],
        },
        "observation": {
            "space": describe_space(env.observation_space),
            "layout": build_observation_layout(obs_components),
        },
        "simulation": {
            **grouped_config["simulation"],
            "render_modes": list(env.metadata.get("render_modes", [])),
            "render_fps": int(env.metadata.get("render_fps", 0)),
        },
        "task": {
            **grouped_config["task"],
            "object_names": list(env.object_names),
        },
    }
