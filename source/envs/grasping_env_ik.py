from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
import imageio

from inverse_kinematics import MyCobotIK

ARM_JOINT_NAMES = (
    "link2_to_link1",
    "link3_to_link2",
    "link4_to_link3",
    "link5_to_link4",
    "link6_to_link5",
    "link6output_to_link6",
)

# Kalau nama actuator arm di XML kamu sama dengan nama joint, ini aman.
ARM_ACTUATOR_NAMES = ARM_JOINT_NAMES

GRIPPER_JOINT_NAMES = (
    "Slider_10",
    "Slider_11",
)

GRIPPER_ACTUATOR_NAMES = (
    "gripper_l",
    "gripper_r",
)

EE_SITE_NAME = "attachment_site"

OBJECT_BODY_NAME = "obj_box"
OBJECT_JOINT_NAME = "obj_box_joint"
OBJECT_SITE_NAME = "obj_box_ref"


# HELPERS
def normalize_quat(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat)

    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    return quat / norm


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Quaternion multiplication.
    Format quaternion: [w, x, y, z]
    """
    w1, x1, y1, z1 = np.asarray(q1, dtype=np.float64)
    w2, x2, y2, z2 = np.asarray(q2, dtype=np.float64)

    return normalize_quat(
        np.array(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dtype=np.float64,
        )
    )


def quat_from_yaw(yaw: float) -> np.ndarray:
    """
    Quaternion rotasi terhadap sumbu Z dunia.
    Format: [w, x, y, z]
    """
    half = 0.5 * float(yaw)

    return normalize_quat(
        np.array(
            [
                np.cos(half),
                0.0,
                0.0,
                np.sin(half),
            ],
            dtype=np.float64,
        )
    )


def wrap_to_pi(angle: float) -> float:
    """
    Membatasi angle ke range [-pi, pi].
    """
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


@dataclass(frozen=True)
class RewardWeights:
    # Semakin dekat EE ke objek, reward makin tinggi.
    distance: float = 2.0

    # Bonus jika EE sudah cukup dekat dengan objek.
    near_object_bonus: float = 5.0

    # Reward bertahap berdasarkan tinggi objek.
    lift_dense: float = 20.0

    # Bonus jika objek sudah terangkat melewati threshold.
    lift_bonus: float = 10.0

    # Penalti kecil agar action tidak terlalu agresif.
    action_penalty: float = 0.01


class GraspingEnvIK(gym.Env):
    """
    Environment awal untuk myCobot grasping menggunakan MuJoCo + IK.

    Action:
        action[0] = dx end-effector
        action[1] = dy end-effector
        action[2] = dz end-effector
        action[3] = gripper action

    Reward:
        reward = -distance(EE, object)
                 + bonus jika EE dekat objek
                 + reward jika objek terangkat
                 + bonus jika objek terangkat cukup tinggi
                 - action penalty

    Tidak menggunakan:
        - contact reward
        - progress reward
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 60,
    }

    def __init__(
        self,
        xml_path: str | Path,
        render_mode: str | None = None,
        max_episode_steps: int = 150,
        frame_skip: int = 10,
        action_scale: float = 0.02,
        close_distance_threshold: float = 0.04,
        lift_height: float = 0.05,
        success_stable_steps: int = 10,
        reward_weights: RewardWeights | None = None,
        randomize_object: bool = True,
        yaw_action_scale: float = np.deg2rad(10.0),
        randomize_object_yaw: bool = True,
    ) -> None:
        super().__init__()

        self.xml_path = Path(xml_path).expanduser().resolve()
        if not self.xml_path.exists():
            raise FileNotFoundError(f"XML file tidak ditemukan: {self.xml_path}")

        self.render_mode = render_mode
        self.max_episode_steps = int(max_episode_steps)
        self.frame_skip = int(frame_skip)
        self.action_scale = float(action_scale)

        self.close_distance_threshold = float(close_distance_threshold)
        self.lift_height = float(lift_height)
        self.success_stable_steps = int(success_stable_steps)
        self.reward_weights = (
            reward_weights if reward_weights is not None else RewardWeights()
        )
        self.randomize_object = bool(randomize_object)

        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)

        self.viewer = None
        self.renderer = None

        # IK solver.
        self.ik = MyCobotIK(
            xml_file=self.xml_path,
            ee_site_name=EE_SITE_NAME,
            joint_names=ARM_JOINT_NAMES,
        )

        # IDs.
        self.ee_site_id = self._site_id(EE_SITE_NAME)
        self.object_site_id = self._site_id(OBJECT_SITE_NAME)
        self.object_body_id = self._body_id(OBJECT_BODY_NAME)
        self.object_joint_id = self._joint_id(OBJECT_JOINT_NAME)

        self.arm_joint_ids = [self._joint_id(name) for name in ARM_JOINT_NAMES]
        self.arm_qpos_ids = [
            int(self.model.jnt_qposadr[jid]) for jid in self.arm_joint_ids
        ]
        self.arm_qvel_ids = [
            int(self.model.jnt_dofadr[jid]) for jid in self.arm_joint_ids
        ]

        self.gripper_joint_ids = [self._joint_id(name) for name in GRIPPER_JOINT_NAMES]
        self.gripper_qpos_ids = [
            int(self.model.jnt_qposadr[jid]) for jid in self.gripper_joint_ids
        ]
        self.gripper_qvel_ids = [
            int(self.model.jnt_dofadr[jid]) for jid in self.gripper_joint_ids
        ]

        self.arm_actuator_ids = [self._actuator_id(name) for name in ARM_ACTUATOR_NAMES]
        self.gripper_actuator_ids = [
            self._actuator_id(name) for name in GRIPPER_ACTUATOR_NAMES
        ]

        # Object free joint qpos address.
        self.object_qpos_addr = int(self.model.jnt_qposadr[self.object_joint_id])
        self.object_qvel_addr = int(self.model.jnt_dofadr[self.object_joint_id])

        # Joint limits.
        self.arm_lower_limits = np.array(
            [self.model.jnt_range[jid][0] for jid in self.arm_joint_ids],
            dtype=np.float64,
        )
        self.arm_upper_limits = np.array(
            [self.model.jnt_range[jid][1] for jid in self.arm_joint_ids],
            dtype=np.float64,
        )

        # Workspace batas target EE.
        # Silakan tuning sesuai workspace real myCobot kamu.
        self.workspace_bounds = np.array(
            [
                [0.10, 0.35],  # x min, x max
                [-0.20, 0.20],  # y min, y max
                [0.03, 0.35],  # z min, z max
            ],
            dtype=np.float64,
        )

        # Home pose awal robot.
        # Silakan tuning kalau pose ini kurang cocok dengan XML kamu.
        self.home_arm_qpos = np.array(
            [0.0, -0.60, 0.85, 0.0, 0.65, 0.0],
            dtype=np.float64,
        )
        self.home_arm_qpos = self._clip_arm_qpos(self.home_arm_qpos)

        # Gripper convention:
        # left = 0.01, right = -0.01  -> open
        # left = -0.02, right = 0.02  -> close
        self.gripper_open_qpos = np.array([0.01, -0.01], dtype=np.float64)
        self.gripper_close_qpos = np.array([-0.02, 0.02], dtype=np.float64)

        # Action: dx, dy, dz, gripper.
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(5,),
            dtype=np.float32,
        )

        # Observation:
        # arm_qpos             6
        # arm_qvel             6
        # gripper_qpos         2
        # gripper_qvel         2
        # ee_pos               3
        # object_pos           3
        # object_quat          4
        # object_linvel        3
        # relative obj - ee    3
        # gripper_opening      1
        # sin and cos yaw      2
        # Total = 35
        self.obs_dim = 35
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )

        self.step_count = 0
        self.success_counter = 0
        self.initial_object_z = 0.015

        self.last_ik_success = False
        self.last_ik_error_norm = np.inf
        self.last_target_ee_pos = np.zeros(3, dtype=np.float64)

        self.yaw_action_scale = float(yaw_action_scale)
        self.randomize_object_yaw = bool(randomize_object_yaw)

        self.use_fixed_down_orientation = True
        self.fixed_down_quat = np.array(
            [1.0, 0.0, 0.0, 0.0],
            dtype=np.float64,
        )

        self.ee_yaw = 0.0

    # ============================================================
    # Gymnasium API
    # ============================================================

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        self.step_count = 0
        self.success_counter = 0
        self.last_ik_success = False
        self.last_ik_error_norm = np.inf
        self.last_target_ee_pos = np.zeros(3, dtype=np.float64)

        mujoco.mj_resetData(self.model, self.data)

        # Reset arm.
        for qpos_id, q in zip(self.arm_qpos_ids, self.home_arm_qpos):
            self.data.qpos[qpos_id] = float(q)

        for qvel_id in self.arm_qvel_ids:
            self.data.qvel[qvel_id] = 0.0

        # Reset gripper open.
        for qpos_id, q in zip(self.gripper_qpos_ids, self.gripper_open_qpos):
            self.data.qpos[qpos_id] = float(q)

        for qvel_id in self.gripper_qvel_ids:
            self.data.qvel[qvel_id] = 0.0

        # Set actuator control awal.
        for act_id, q in zip(self.arm_actuator_ids, self.home_arm_qpos):
            self.data.ctrl[act_id] = float(q)

        for act_id, q in zip(self.gripper_actuator_ids, self.gripper_open_qpos):
            self.data.ctrl[act_id] = float(q)

        # Reset object.
        object_pos = self._sample_object_position()

        if self.randomize_object_yaw:
            object_yaw = self.np_random.uniform(-np.pi / 4, np.pi / 4)
        else:
            object_yaw = 0.0

        object_quat = quat_from_yaw(object_yaw)

        self._set_object_pose(
            pos=object_pos,
            quat=object_quat,
        )

        # Reset target yaw EE.
        # Untuk awal boleh 0.0.
        # Nanti bisa juga di-random jika ingin curriculum lebih sulit.
        self.ee_yaw = 0.0

        # Forward simulation.
        mujoco.mj_forward(self.model, self.data)

        # Stabilize sebentar.
        for _ in range(20):
            mujoco.mj_step(self.model, self.data)

        self.initial_object_z = float(self._object_pos()[2])

        obs = self._get_obs()

        distance = float(np.linalg.norm(self._object_pos() - self._ee_pos()))
        lift_amount = float(max(0.0, self._object_pos()[2] - self.initial_object_z))

        info = {
            "distance_ee_object": distance,
            "initial_object_z": self.initial_object_z,
            "lift_amount": lift_amount,
            "is_success": self._is_success(distance=distance, lift_amount=lift_amount),
        }

        if self.render_mode == "human":
            self.render()

        return obs, info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        self.step_count += 1

        action = np.asarray(action, dtype=np.float64)
        action = np.clip(action, -1.0, 1.0)

        delta_pos = action[:3] * self.action_scale

        delta_yaw = float(action[3]) * self.yaw_action_scale
        self.ee_yaw = wrap_to_pi(self.ee_yaw + delta_yaw)

        gripper_action = float(action[4])

        current_q = self._arm_qpos()
        current_ee_pos = self._ee_pos()
        target_ee_pos = current_ee_pos + delta_pos
        target_ee_pos = self._clip_workspace(target_ee_pos)

        self.last_target_ee_pos = target_ee_pos.copy()

        ik_failed = False

        # Solve IK.
        # Prefer solve_delta() kalau ada di inverse_kinematics_fixed.py.
        # Kalau tidak ada, fallback ke solve().
        try:
            target_quat = self._target_ee_quat()

            if hasattr(self.ik, "sync_base_from_data"):
                self.ik.sync_base_from_data(
                    qpos=self.data.qpos.copy(),
                    qvel=self.data.qvel.copy(),
                )

            result = self.ik.solve(
                target_pos=target_ee_pos,
                target_quat=target_quat,
                initial_q=current_q,
                position_only=False,
                max_iters=60,
                position_tolerance=2e-3,
                rotation_tolerance=np.deg2rad(10.0),
                damping=1e-3,
                step_size=0.4,
                max_delta=np.deg2rad(5.0),
                rotation_weight=0.35,
                random_restarts=0,
                seed=None,
            )

            q_goal = self._clip_arm_qpos(result.q_rad)
            self.last_ik_success = bool(result.success)
            self.last_ik_error_norm = float(result.position_error_norm)

            if not np.all(np.isfinite(q_goal)):
                ik_failed = True
                q_goal = current_q.copy()

        except Exception:
            ik_failed = True
            self.last_ik_success = False
            self.last_ik_error_norm = np.inf
            q_goal = current_q.copy()

        # Apply arm target ke position actuator.
        for act_id, q in zip(self.arm_actuator_ids, q_goal):
            self.data.ctrl[act_id] = float(q)

        # Apply gripper.
        self._apply_gripper(gripper_action)

        # Simulate beberapa physics step untuk 1 action RL.
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        reward, info = self._compute_reward(action=action, ik_failed=ik_failed)

        if info["is_success"]:
            self.success_counter += 1
        else:
            self.success_counter = 0

        terminated = self.success_counter >= self.success_stable_steps
        truncated = self.step_count >= self.max_episode_steps

        obs = self._get_obs()

        info["step_count"] = self.step_count
        info["success_counter"] = self.success_counter
        info["terminated"] = terminated
        info["truncated"] = truncated

        if self.render_mode == "human":
            self.render()

        return obs, float(reward), bool(terminated), bool(truncated), info

    # ============================================================
    # Reward
    # ============================================================

    def _compute_reward(
        self,
        action: np.ndarray,
        ik_failed: bool,
    ) -> tuple[float, dict]:
        """
        Reward sederhana untuk tahap awal.

        Komponen:
        1. Distance reward:
           EE semakin dekat ke objek, reward semakin baik.

        2. Near object bonus:
           Bonus jika EE sudah berada cukup dekat dengan objek.

        3. Lift dense reward:
           Reward bertahap jika objek mulai naik dari posisi awal.

        4. Lift bonus:
           Bonus besar jika objek sudah terangkat melewati threshold.

        5. Action penalty:
           Penalti kecil agar action tidak terlalu kasar.

        Tidak ada:
        - progress reward
        - contact reward
        """

        weights = self.reward_weights

        ee_pos = self._ee_pos()
        obj_pos = self._object_pos()

        distance = float(np.linalg.norm(obj_pos - ee_pos))
        close_to_object = distance <= self.close_distance_threshold

        lift_amount = float(max(0.0, obj_pos[2] - self.initial_object_z))
        lifted = lift_amount >= self.lift_height

        action_norm_sq = float(np.sum(np.square(action)))

        distance_reward = -weights.distance * distance

        near_bonus = weights.near_object_bonus if close_to_object else 0.0

        lift_dense_reward = weights.lift_dense * lift_amount

        lift_bonus = weights.lift_bonus if lifted else 0.0

        action_penalty = -weights.action_penalty * action_norm_sq

        reward = (
            distance_reward
            + near_bonus
            + lift_dense_reward
            + lift_bonus
            + action_penalty
        )

        is_success = self._is_success(
            distance=distance,
            lift_amount=lift_amount,
        )

        info = {
            "reward_total": float(reward),
            "reward_distance": float(distance_reward),
            "reward_near_bonus": float(near_bonus),
            "reward_lift_dense": float(lift_dense_reward),
            "reward_lift_bonus": float(lift_bonus),
            "reward_action_penalty": float(action_penalty),
            "distance_ee_object": distance,
            "close_to_object": bool(close_to_object),
            "close_distance_threshold": float(self.close_distance_threshold),
            "object_z": float(obj_pos[2]),
            "initial_object_z": float(self.initial_object_z),
            "lift_amount": float(lift_amount),
            "lift_height": float(self.lift_height),
            "lifted": bool(lifted),
            "action_norm_sq": float(action_norm_sq),
            "ik_failed": bool(ik_failed),
            "ik_success": bool(self.last_ik_success),
            "ik_error_norm": float(self.last_ik_error_norm),
            "target_ee_pos": self.last_target_ee_pos.copy(),
            "is_success": bool(is_success),
        }

        return float(reward), info

    def _is_success(
        self,
        *,
        distance: float | None = None,
        lift_amount: float | None = None,
    ) -> bool:
        """
        Success jika:
        1. EE sudah dekat dengan objek
        2. Objek sudah terangkat minimal lift_height
        """

        if distance is None:
            distance = float(np.linalg.norm(self._object_pos() - self._ee_pos()))

        if lift_amount is None:
            obj_pos = self._object_pos()
            lift_amount = float(max(0.0, obj_pos[2] - self.initial_object_z))

        close_to_object = distance <= self.close_distance_threshold
        lifted = lift_amount >= self.lift_height

        return bool(close_to_object and lifted)

    # ============================================================
    # Observation
    # ============================================================

    def _get_obs(self) -> np.ndarray:
        arm_qpos = self._arm_qpos()
        arm_qvel = self._arm_qvel()

        gripper_qpos = self._gripper_qpos()
        gripper_qvel = self._gripper_qvel()

        ee_pos = self._ee_pos()

        obj_pos = self._object_pos()
        obj_quat = self._object_quat()
        obj_linvel = self._object_linvel()

        relative_pos = obj_pos - ee_pos

        gripper_opening = np.array(
            [abs(gripper_qpos[1] - gripper_qpos[0])],
            dtype=np.float64,
        )

        ee_yaw_obs = np.array(
            [
                np.sin(self.ee_yaw),
                np.cos(self.ee_yaw),
            ],
            dtype=np.float64,
        )

        obs = np.concatenate(
            [
                arm_qpos,
                arm_qvel,
                gripper_qpos,
                gripper_qvel,
                ee_pos,
                obj_pos,
                obj_quat,
                obj_linvel,
                relative_pos,
                gripper_opening,
                ee_yaw_obs,
            ],
            dtype=np.float64,
        )

        if obs.shape[0] != self.obs_dim:
            raise RuntimeError(
                f"Observation dim salah. Expected {self.obs_dim}, got {obs.shape[0]}"
            )

        return obs.astype(np.float32)

    # ============================================================
    # State helpers
    # ============================================================

    def _arm_qpos(self) -> np.ndarray:
        return self.data.qpos[self.arm_qpos_ids].copy()

    def _arm_qvel(self) -> np.ndarray:
        return self.data.qvel[self.arm_qvel_ids].copy()

    def _gripper_qpos(self) -> np.ndarray:
        return self.data.qpos[self.gripper_qpos_ids].copy()

    def _gripper_qvel(self) -> np.ndarray:
        return self.data.qvel[self.gripper_qvel_ids].copy()

    def _ee_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.ee_site_id].copy()

    def _object_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.object_site_id].copy()

    def _object_quat(self) -> np.ndarray:
        # Free joint qpos layout:
        # qpos[addr + 0: addr + 3] = position
        # qpos[addr + 3: addr + 7] = quaternion wxyz
        quat = self.data.qpos[
            self.object_qpos_addr + 3 : self.object_qpos_addr + 7
        ].copy()
        norm = np.linalg.norm(quat)

        if norm < 1e-12:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

        return quat / norm

    def _object_linvel(self) -> np.ndarray:
        # Free joint qvel layout:
        # qvel[addr + 0: addr + 3] = linear velocity
        # qvel[addr + 3: addr + 6] = angular velocity
        return self.data.qvel[self.object_qvel_addr : self.object_qvel_addr + 3].copy()

    def _target_ee_quat(self) -> np.ndarray:
        """
        Membuat target quaternion EE.

        Prinsip:
        - fixed_down_quat membuat EE menghadap bawah.
        - yaw_quat memutar EE di sekitar sumbu Z dunia.
        - Hasil akhirnya: EE tetap menghadap bawah, tapi yaw berubah.
        """
        yaw_quat = quat_from_yaw(self.ee_yaw)

        target_quat = quat_multiply(
            yaw_quat,
            self.fixed_down_quat,
        )

        return normalize_quat(target_quat)

    # ============================================================
    # Control helpers
    # ============================================================

    def _apply_gripper(self, gripper_action: float) -> None:
        """
        gripper_action:
            -1 = open
            +1 = close
        """

        g = float(np.clip(gripper_action, -1.0, 1.0))

        left_ctrl = np.interp(
            g,
            [-1.0, 1.0],
            [self.gripper_open_qpos[0], self.gripper_close_qpos[0]],
        )

        right_ctrl = np.interp(
            g,
            [-1.0, 1.0],
            [self.gripper_open_qpos[1], self.gripper_close_qpos[1]],
        )

        self.data.ctrl[self.gripper_actuator_ids[0]] = float(left_ctrl)
        self.data.ctrl[self.gripper_actuator_ids[1]] = float(right_ctrl)

    def _clip_arm_qpos(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64).reshape(6)
        return np.clip(q, self.arm_lower_limits, self.arm_upper_limits)

    def _clip_workspace(self, pos: np.ndarray) -> np.ndarray:
        pos = np.asarray(pos, dtype=np.float64).reshape(3)
        clipped = pos.copy()

        clipped[0] = np.clip(
            clipped[0], self.workspace_bounds[0, 0], self.workspace_bounds[0, 1]
        )
        clipped[1] = np.clip(
            clipped[1], self.workspace_bounds[1, 0], self.workspace_bounds[1, 1]
        )
        clipped[2] = np.clip(
            clipped[2], self.workspace_bounds[2, 0], self.workspace_bounds[2, 1]
        )

        return clipped

    # ============================================================
    # Object reset helpers
    # ============================================================

    def _sample_object_position(self) -> np.ndarray:
        if not self.randomize_object:
            return np.array([0.20, 0.00, 0.015], dtype=np.float64)

        x = self.np_random.uniform(0.15, 0.25)
        y = self.np_random.uniform(-0.15, 0.15)
        z = 0.015

        return np.array([x, y, z], dtype=np.float64)

    def _set_object_pose(self, pos: np.ndarray, quat: np.ndarray) -> None:
        pos = np.asarray(pos, dtype=np.float64).reshape(3)
        quat = np.asarray(quat, dtype=np.float64).reshape(4)

        quat_norm = np.linalg.norm(quat)
        if quat_norm < 1e-12:
            quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        else:
            quat = quat / quat_norm

        addr = self.object_qpos_addr

        self.data.qpos[addr : addr + 3] = pos
        self.data.qpos[addr + 3 : addr + 7] = quat

        self.data.qvel[self.object_qvel_addr : self.object_qvel_addr + 6] = 0.0

    # ============================================================
    # MuJoCo name helpers
    # ============================================================

    def _joint_id(self, name: str) -> int:
        joint_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            name,
        )

        if joint_id < 0:
            raise ValueError(f"Joint tidak ditemukan di XML: {name}")

        return int(joint_id)

    def _actuator_id(self, name: str) -> int:
        actuator_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            name,
        )

        if actuator_id < 0:
            raise ValueError(f"Actuator tidak ditemukan di XML: {name}")

        return int(actuator_id)

    def _site_id(self, name: str) -> int:
        site_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_SITE,
            name,
        )

        if site_id < 0:
            raise ValueError(f"Site tidak ditemukan di XML: {name}")

        return int(site_id)

    def _body_id(self, name: str) -> int:
        body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            name,
        )

        if body_id < 0:
            raise ValueError(f"Body tidak ditemukan di XML: {name}")

        return int(body_id)

    # ============================================================
    # Rendering
    # ============================================================

    def render(self):
        if self.render_mode == "rgb_array":
            if self.renderer is None:
                self.renderer = mujoco.Renderer(
                    self.model,
                    height=480,
                    width=640,
                )

            self.renderer.update_scene(
                self.data,
                camera="record_camera",
            )
            return self.renderer.render()

        if self.render_mode == "human":
            if self.viewer is None:
                from mujoco import viewer as mujoco_viewer

                self.viewer = mujoco_viewer.launch_passive(
                    self.model,
                    self.data,
                )

            if self.viewer.is_running():
                self.viewer.sync()

            return None

        return None

    def close(self) -> None:
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

        self.renderer = None
