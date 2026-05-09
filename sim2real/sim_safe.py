from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as R
from stable_baselines3 import SAC

from sim2real.remote import MyCobotRemote
from sim2real.vision import AprilTagPose

# ------------------------------------------------------------------
# Defaults
# ------------------------------------------------------------------

DEFAULT_ROBOT_IP = "10.244.4.108"
DEFAULT_MODEL_PATH = (
    "/home/fauzan/Grasping_Skripsi/source/envs/sac_lift_1450000_steps.zip"
)
DEFAULT_CAM_INDEX = 2
DEFAULT_BASE_TAG_ID = 6
DEFAULT_OBJ_TAG_ID = 0
DEFAULT_TARGET_POS = np.array([0.18, 0.0, 0.15], dtype=np.float64)

# MuJoCo actuator limits from source/robot/robot.xml
JOINT_LIMITS_DEG = np.rad2deg(
    np.array(
        [
            [-2.9321, 2.9321],
            [-2.3561, 2.3561],
            [-2.6179, 2.6179],
            [-2.5307, 2.5307],
            [-2.8797, 2.8797],
            [-3.1416, 3.1416],
        ],
        dtype=np.float64,
    )
)


@dataclass
class SafetyConfig:
    action_scale_rad: float = 0.01
    action_clip: float = 1.0
    move_speed: int = 20
    loop_dt: float = 0.05
    ack_timeout: float = 5.0
    settle_timeout: float = 20.0
    poll_dt: float = 0.2
    joint_tolerance_deg: float = 1.5
    stable_polls_required: int = 3
    min_command_delta_deg: float = 0.15
    max_step_deg: float = 3.0
    max_consecutive_failures: int = 5
    show_window: bool = True
    object_z_offset_m: float = 0.0
    target_lift_height_m: float = 0.10
    grasp_close_distance_m: float = 0.015
    grasp_release_distance_m: float = 0.055
    grasp_close_angle_deg: float = 25.0


class StateTracker:
    def __init__(self):
        self.obj_pos = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.obj_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.has_object_pose = False
        self.initial_obj_pos = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.has_initial_obj_pos = False
        self.grasp_latched = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Safer sim2real controller for MyCobot with blocking action gate."
    )
    parser.add_argument("--robot-ip", default=DEFAULT_ROBOT_IP)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--cam-index", type=int, default=DEFAULT_CAM_INDEX)
    parser.add_argument("--base-tag-id", type=int, default=DEFAULT_BASE_TAG_ID)
    parser.add_argument("--obj-tag-id", type=int, default=DEFAULT_OBJ_TAG_ID)
    parser.add_argument("--move-speed", type=int, default=20)
    parser.add_argument("--ack-timeout", type=float, default=5.0)
    parser.add_argument("--settle-timeout", type=float, default=20.0)
    parser.add_argument("--poll-dt", type=float, default=0.2)
    parser.add_argument("--loop-dt", type=float, default=0.05)
    parser.add_argument("--joint-tolerance-deg", type=float, default=1.5)
    parser.add_argument("--stable-polls", type=int, default=3)
    parser.add_argument("--max-step-deg", type=float, default=3.0)
    parser.add_argument("--min-command-delta-deg", type=float, default=0.15)
    parser.add_argument("--action-clip", type=float, default=1.0)
    parser.add_argument("--object-z-offset-m", type=float, default=0.0)
    parser.add_argument(
        "--hide-window",
        action="store_true",
        help="Disable OpenCV preview window from AprilTag vision.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> SafetyConfig:
    return SafetyConfig(
        move_speed=int(args.move_speed),
        loop_dt=float(args.loop_dt),
        ack_timeout=float(args.ack_timeout),
        settle_timeout=float(args.settle_timeout),
        poll_dt=float(args.poll_dt),
        joint_tolerance_deg=float(args.joint_tolerance_deg),
        stable_polls_required=max(1, int(args.stable_polls)),
        max_step_deg=float(args.max_step_deg),
        min_command_delta_deg=float(args.min_command_delta_deg),
        action_clip=float(args.action_clip),
        show_window=not bool(args.hide_window),
        object_z_offset_m=float(args.object_z_offset_m),
    )


def _scipy_quat_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
    return np.array(
        [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float64
    )


def _normalize_quat(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return quat / norm


def _quat_conjugate(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    return np.array([quat[0], -quat[1], -quat[2], -quat[3]], dtype=np.float64)


def _quat_multiply(quat_a: np.ndarray, quat_b: np.ndarray) -> np.ndarray:
    wa, xa, ya, za = np.asarray(quat_a, dtype=np.float64)
    wb, xb, yb, zb = np.asarray(quat_b, dtype=np.float64)
    return np.array(
        [
            wa * wb - xa * xb - ya * yb - za * zb,
            wa * xb + xa * wb + ya * zb - za * yb,
            wa * yb - xa * zb + ya * wb + za * xb,
            wa * zb + xa * yb - ya * xb + za * wb,
        ],
        dtype=np.float64,
    )


def _rotation_vector(source_quat: np.ndarray, target_quat: np.ndarray) -> np.ndarray:
    source_quat = _normalize_quat(source_quat)
    target_quat = _normalize_quat(target_quat)
    delta = _quat_multiply(target_quat, _quat_conjugate(source_quat))
    delta = _normalize_quat(delta)
    if delta[0] < 0.0:
        delta = -delta

    xyz = delta[1:]
    sin_half = np.linalg.norm(xyz)
    if sin_half < 1e-12:
        return np.zeros(3, dtype=np.float64)

    angle = 2.0 * np.arctan2(sin_half, np.clip(delta[0], -1.0, 1.0))
    axis = xyz / sin_half
    return axis * angle


def _get_pose_error(
    source_pos: np.ndarray,
    source_quat: np.ndarray,
    target_pos: np.ndarray,
    target_quat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    pos_error = np.asarray(target_pos, dtype=np.float64) - np.asarray(
        source_pos, dtype=np.float64
    )
    rot_error = _rotation_vector(source_quat, target_quat)
    return pos_error, rot_error


def wrap_joint_error_deg(current_deg: np.ndarray, target_deg: np.ndarray) -> np.ndarray:
    return (current_deg - target_deg + 180.0) % 360.0 - 180.0


def clip_joint_targets_deg(target_deg: np.ndarray) -> np.ndarray:
    lower = JOINT_LIMITS_DEG[:, 0]
    upper = JOINT_LIMITS_DEG[:, 1]
    return np.clip(target_deg, lower, upper)


def is_valid_robot_state(values: np.ndarray, expected_size: int) -> bool:
    return values.shape == (expected_size,) and np.all(np.isfinite(values))


def compute_safe_target_angles_deg(
    current_angles_deg: np.ndarray, action: np.ndarray, cfg: SafetyConfig
) -> np.ndarray:
    action = np.asarray(action[:6], dtype=np.float64)
    action = np.clip(action, -cfg.action_clip, cfg.action_clip)

    delta_deg = np.rad2deg(action * cfg.action_scale_rad)
    delta_deg = np.clip(delta_deg, -cfg.max_step_deg, cfg.max_step_deg)

    target_deg = current_angles_deg + delta_deg
    return clip_joint_targets_deg(target_deg)


def build_observation(
    mc: MyCobotRemote,
    vision: AprilTagPose,
    state: StateTracker,
    obj_tag_id: int,
    cfg: SafetyConfig,
) -> np.ndarray | None:
    arm_qpos = np.deg2rad(np.asarray(mc.angles, dtype=np.float64))
    gripper_state = np.array([0.02, -0.02], dtype=np.float64)
    robot_qpos = np.concatenate([arm_qpos, gripper_state])
    robot_qvel = np.zeros(8, dtype=np.float64)

    tags, _ = vision.get_tag_poses(show_window=cfg.show_window)
    if obj_tag_id in tags:
        obj_pos = np.asarray(tags[obj_tag_id]["pos"], dtype=np.float64).copy()
        obj_pos[2] += cfg.object_z_offset_m
        obj_rpy = np.asarray(tags[obj_tag_id]["rpy"], dtype=np.float64)
        obj_quat = _scipy_quat_to_wxyz(
            R.from_euler("xyz", obj_rpy, degrees=True).as_quat()
        )
        state.obj_pos = obj_pos
        state.obj_quat = obj_quat
        state.has_object_pose = True
        if not state.has_initial_obj_pos:
            state.initial_obj_pos = obj_pos.copy()
            state.has_initial_obj_pos = True
    elif state.has_object_pose:
        obj_pos = state.obj_pos
        obj_quat = state.obj_quat
    else:
        return None

    ee_pos = np.asarray(mc.coords[:3], dtype=np.float64) / 1000.0
    ee_rpy = np.deg2rad(np.asarray(mc.coords[3:], dtype=np.float64))
    ee_quat = _scipy_quat_to_wxyz(R.from_euler("xyz", ee_rpy).as_quat())

    if not state.has_initial_obj_pos:
        return None

    target_pos = state.initial_obj_pos + np.array(
        [0.0, 0.0, cfg.target_lift_height_m], dtype=np.float64
    )
    target_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    ee_obj_pos_error, ee_obj_rot_error = _get_pose_error(
        ee_pos, ee_quat, obj_pos, obj_quat
    )
    obj_target_pos_error, _ = _get_pose_error(
        obj_pos, obj_quat, target_pos, target_quat
    )

    ee_obj_dist = float(np.linalg.norm(ee_obj_pos_error))
    ee_obj_angle = float(np.linalg.norm(ee_obj_rot_error))
    should_close = (
        ee_obj_dist < cfg.grasp_close_distance_m
        and ee_obj_angle < np.deg2rad(cfg.grasp_close_angle_deg)
    )
    keep_closed = state.grasp_latched and ee_obj_dist < cfg.grasp_release_distance_m
    state.grasp_latched = bool(should_close or keep_closed)

    lift_height = float(obj_pos[2] - state.initial_obj_pos[2])

    obs = np.concatenate(
        [
            robot_qpos,
            robot_qvel,
            gripper_state,
            ee_pos,
            ee_quat,
            obj_pos,
            obj_quat,
            target_pos,
            target_quat,
            ee_obj_pos_error,
            ee_obj_rot_error,
            obj_target_pos_error,
            np.array(
                [
                    np.linalg.norm(ee_obj_pos_error),
                    np.linalg.norm(ee_obj_rot_error),
                    np.linalg.norm(obj_target_pos_error),
                    lift_height,
                    float(state.grasp_latched),
                ],
                dtype=np.float64,
            ),
        ]
    ).astype(np.float32)

    if not np.all(np.isfinite(obs)):
        return None
    return obs


def wait_until_target_stable(
    mc: MyCobotRemote, target_deg: np.ndarray, cfg: SafetyConfig
) -> tuple[bool, float]:
    deadline = time.monotonic() + cfg.settle_timeout
    stable_polls = 0
    last_max_error = float("inf")

    while time.monotonic() <= deadline:
        if not mc.update_state():
            stable_polls = 0
            time.sleep(cfg.poll_dt)
            continue

        current_deg = np.asarray(mc.angles, dtype=np.float64)
        if not is_valid_robot_state(current_deg, mc.NUM_JOINTS):
            stable_polls = 0
            time.sleep(cfg.poll_dt)
            continue

        joint_errors = wrap_joint_error_deg(current_deg, target_deg)
        last_max_error = float(np.max(np.abs(joint_errors)))

        if last_max_error <= cfg.joint_tolerance_deg:
            stable_polls += 1
            if stable_polls >= cfg.stable_polls_required:
                return True, last_max_error
        else:
            stable_polls = 0

        time.sleep(cfg.poll_dt)

    return False, last_max_error


def main():
    args = parse_args()
    cfg = build_config(args)

    mc = MyCobotRemote(args.robot_ip)
    model = SAC.load(args.model_path)
    vision = AprilTagPose(base_id=args.base_tag_id, cam_index=args.cam_index)
    state = StateTracker()

    consecutive_failures = 0

    try:
        mc.power_on()
        time.sleep(2.0)
        print("Safer sim2real controller started. Press Ctrl-C to stop.")

        while True:
            if not mc.update_state():
                consecutive_failures += 1
                print(f"State update failed ({consecutive_failures}).")
                if consecutive_failures >= cfg.max_consecutive_failures:
                    raise RuntimeError("Too many robot state failures.")
                time.sleep(cfg.loop_dt)
                continue

            current_angles_deg = np.asarray(mc.angles, dtype=np.float64)
            current_coords = np.asarray(mc.coords, dtype=np.float64)
            if not is_valid_robot_state(current_angles_deg, mc.NUM_JOINTS):
                consecutive_failures += 1
                print("Invalid joint state received. Skipping cycle.")
                time.sleep(cfg.loop_dt)
                continue
            if not is_valid_robot_state(current_coords, mc.NUM_JOINTS):
                consecutive_failures += 1
                print("Invalid Cartesian state received. Skipping cycle.")
                time.sleep(cfg.loop_dt)
                continue

            obs = build_observation(
                mc=mc,
                vision=vision,
                state=state,
                obj_tag_id=args.obj_tag_id,
                cfg=cfg,
            )
            if obs is None:
                consecutive_failures += 1
                print("Observation incomplete. Waiting for a valid object pose.")
                time.sleep(cfg.loop_dt)
                continue

            action, _ = model.predict(obs, deterministic=True)
            target_deg = compute_safe_target_angles_deg(current_angles_deg, action, cfg)
            command_delta = float(np.max(np.abs(target_deg - current_angles_deg)))

            if command_delta < cfg.min_command_delta_deg:
                consecutive_failures = 0
                time.sleep(cfg.loop_dt)
                continue

            ack_ok = mc.send_angles(
                target_deg.tolist(),
                cfg.move_speed,
                wait=False,
                ack_timeout=cfg.ack_timeout,
            )
            if not ack_ok:
                consecutive_failures += 1
                print(f"SET_ANGLES ACK failed ({consecutive_failures}).")
                if consecutive_failures >= cfg.max_consecutive_failures:
                    raise RuntimeError("Too many command ACK failures.")
                time.sleep(cfg.loop_dt)
                continue

            settled, max_error = wait_until_target_stable(mc, target_deg, cfg)
            if not settled:
                consecutive_failures += 1
                print(
                    "Target not settled before timeout. "
                    f"max_error_deg={max_error:.3f} failures={consecutive_failures}"
                )
                if consecutive_failures >= cfg.max_consecutive_failures:
                    raise RuntimeError("Too many settle failures.")
            else:
                consecutive_failures = 0
                print(
                    "Target settled. "
                    f"delta_deg={command_delta:.3f} max_error_deg={max_error:.3f}"
                )

            time.sleep(cfg.loop_dt)

    except KeyboardInterrupt:
        print("Stop signal received.")
    finally:
        mc.stop()
        vision.release()
        print("Safer sim2real controller stopped.")


if __name__ == "__main__":
    main()
