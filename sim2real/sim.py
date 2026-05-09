import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from stable_baselines3 import SAC

from sim2real.remote import MyCobotRemote
from sim2real.vision import AprilTagPose

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

ROBOT_IP = "10.16.121.76"
MODEL_PATH = (
    "/home/fauzan/Mujoco/Skripsi/logs/models/GraspingEnv/"
    "SAC_26_02_2026_14_27_49/sac_lift_800000_steps.zip"
)
CAM_INDEX = 2
BASE_TAG_ID = 12
OBJ_TAG_ID = 1

TARGET_POS = np.array([0.18, 0.0, 0.15])
ACTION_SCALE = 0.01  # rad per unit action
MOVE_SPEED = 20  # MyCobot speed (0–100)
GRIPPER_SPEED = 50
LOOP_DT = 0.05  # seconds between control steps
COMMAND_ACK_TIMEOUT = 0.5
COMMAND_SETTLE_TIMEOUT = 4.0
JOINT_TOLERANCE_DEG = 2.0
STATE_POLL_DT = 0.05


# ------------------------------------------------------------------
# State tracker (fallback values when sensor data is unavailable)
# ------------------------------------------------------------------


class StateTracker:
    def __init__(self):
        self.obj_pos = np.array([0.0, 0.0, 0.0])
        self.obj_quat = np.array([1.0, 0.0, 0.0, 0.0])  # w, x, y, z


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _scipy_quat_to_wxyz(q: np.ndarray) -> np.ndarray:
    """Convert scipy quaternion [x, y, z, w] → [w, x, y, z]."""
    return np.array([q[3], q[0], q[1], q[2]])


def compute_target_angles_deg(
    current_angles_deg: np.ndarray, action: np.ndarray
) -> np.ndarray:
    """Mirror the training env: add the action delta in radians, then convert back."""
    current_angles_rad = np.deg2rad(current_angles_deg)
    target_angles_rad = current_angles_rad + action[:6] * ACTION_SCALE
    return np.rad2deg(target_angles_rad)


def get_observation(
    mc: MyCobotRemote,
    vision: AprilTagPose,
    target_pos: np.ndarray,
    state: StateTracker,
) -> np.ndarray:
    """
    Build a 41-element observation vector from real robot sensors.

    IMPORTANT: mc.update_state() must be called before this function
    so angles and coords come from a single cached UDP response.

    Layout:
        robot_qpos   (8)  — 6 joint angles + 2 gripper fingers
        robot_qvel   (8)  — zeros (not measured)
        gripper_qpos (2)
        obj_pos      (3)
        obj_quat     (4)  — w, x, y, z
        target_pos   (3)
        rel_obj_ee   (3)  — object relative to end-effector
        rel_obj_tgt  (3)  — object relative to target
        ee_pos       (3)
        ee_quat      (4)  — w, x, y, z
                   -----
                   TOTAL = 41
    """
    # --- Robot state (from cache — no extra UDP call) ---
    arm_qpos = np.deg2rad(mc.angles)
    gripper_qpos = np.array([0.02, -0.02])
    robot_qpos = np.concatenate([arm_qpos, gripper_qpos])
    robot_qvel = np.zeros(8)

    # --- Object pose from AprilTag (camera read happens here) ---
    tags, _ = vision.get_tag_poses(show_window=True)  # imshow + waitKey inside
    if OBJ_TAG_ID in tags:
        # print(f"Tag {OBJ_TAG_ID} detected. Updating object pose.")
        obj_pos = tags[OBJ_TAG_ID]["pos"]
        obj_rpy = tags[OBJ_TAG_ID]["rpy"]
        obj_quat = _scipy_quat_to_wxyz(
            R.from_euler("xyz", obj_rpy, degrees=True).as_quat()
        )
        state.obj_pos = obj_pos
        state.obj_pos[2] -= 10
        state.obj_quat = obj_quat
    else:
        # print(f"Tag {OBJ_TAG_ID} NOT detected. Using last known pose.")
        obj_pos = state.obj_pos
        obj_quat = state.obj_quat

    print(f"Object pos: {obj_pos}, quat: {obj_quat}")
    # --- End-effector pose (from cache — no extra UDP call) ---
    ee_pos = np.array(mc.coords[:3]) / 1000.0  # mm → m
    ee_rpy = np.deg2rad(mc.coords[3:])
    ee_quat = _scipy_quat_to_wxyz(R.from_euler("xyz", ee_rpy).as_quat())

    # --- Relative positions ---
    rel_obj_ee = obj_pos - ee_pos
    rel_obj_tgt = obj_pos - target_pos

    return np.concatenate(
        [
            robot_qpos,  # 8
            robot_qvel,  # 8
            gripper_qpos,  # 2
            obj_pos,  # 3
            obj_quat,  # 4
            target_pos,  # 3
            rel_obj_ee,  # 3
            rel_obj_tgt,  # 3
            ee_pos,  # 3
            ee_quat,  # 4
        ]
    ).astype(np.float32)


# ------------------------------------------------------------------
# Main control loop
# ------------------------------------------------------------------


def main():
    mc = MyCobotRemote(ROBOT_IP)
    model = SAC.load(MODEL_PATH)
    vision = AprilTagPose(base_id=BASE_TAG_ID, cam_index=CAM_INDEX)
    state = StateTracker()

    try:
        mc.power_on()
        time.sleep(2)
        print("System started. Press Ctrl-C to stop.")

        while True:
            mc.update_state()

            current_angles_deg = np.asarray(mc.angles, dtype=np.float64)
            current_coords = np.asarray(mc.coords, dtype=np.float64)
            if (
                current_angles_deg.shape != (mc.NUM_JOINTS,)
                or current_coords.shape != (mc.NUM_JOINTS,)
            ):
                continue

            print(f"Current angles (deg): {current_angles_deg.tolist()}")
            obs = get_observation(mc, vision, TARGET_POS, state)
            action, _ = model.predict(obs, deterministic=True)
            print(f"Action: {action}")
            new_angles = compute_target_angles_deg(current_angles_deg, action)

            print(f"Sending angles (deg): {new_angles.tolist()}")
            command_ok = mc.send_angles(
                new_angles.tolist(),
                MOVE_SPEED,
                wait=True,
                tolerance_deg=JOINT_TOLERANCE_DEG,
                ack_timeout=COMMAND_ACK_TIMEOUT,
                settle_timeout=COMMAND_SETTLE_TIMEOUT,
                poll_interval=STATE_POLL_DT,
            )
            if not command_ok:
                print("Warning: target tidak di-ack atau belum tercapai sebelum timeout.")
            # mc.set_gripper_state(1 if action[6] < 0 else 0, GRIPPER_SPEED)

            time.sleep(LOOP_DT)

    except KeyboardInterrupt:
        print("Stop signal received.")
    finally:
        mc.stop()
        vision.release()
        print("Sim2Real test complete.")


if __name__ == "__main__":
    main()
