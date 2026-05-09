from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import mujoco
except ModuleNotFoundError as exc:  # pragma: no cover - depends on runtime env
    mujoco = None
    _MUJOCO_IMPORT_ERROR = exc
else:
    _MUJOCO_IMPORT_ERROR = None


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
_LOCAL_ROBOT_XML = SCRIPT_DIR / "robot.xml"
DEFAULT_XML_PATH = (
    _LOCAL_ROBOT_XML
    if _LOCAL_ROBOT_XML.exists()
    else PROJECT_ROOT / "source" / "robot" / "robot.xml"
)
DEFAULT_EE_SITE_NAME = "attachment_site"
ARM_JOINT_NAMES = (
    "link2_to_link1",
    "link3_to_link2",
    "link4_to_link3",
    "link5_to_link4",
    "link6_to_link5",
    "link6output_to_link6",
)


@dataclass(frozen=True, slots=True)
class IKRuntimeConfig:
    """Parameter IK yang ringan untuk dipakai berulang di environment RL."""

    max_iters: int = 35
    position_tolerance: float = 2e-3
    rotation_tolerance: float = np.deg2rad(5.0)
    damping: float = 1e-3
    step_size: float = 0.5
    max_delta: float = np.deg2rad(5.0)
    rotation_weight: float = 0.2
    random_restarts: int = 0
    seed: int | None = None


RL_FAST_CONFIG = IKRuntimeConfig()


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


def _rotation_vector(current_quat: np.ndarray, target_quat: np.ndarray) -> np.ndarray:
    current_quat = _normalize_quat(current_quat)
    target_quat = _normalize_quat(target_quat)
    delta = _quat_multiply(target_quat, _quat_conjugate(current_quat))
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


def _quat_from_euler_xyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)

    quat = np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=np.float64,
    )
    return _normalize_quat(quat)


def _quat_to_euler_xyz(quat: np.ndarray) -> np.ndarray:
    quat = _normalize_quat(quat)
    w, x, y, z = quat

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw], dtype=np.float64)


def _quat_to_mat(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = _normalize_quat(quat)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _format_values(values: np.ndarray | list[float], precision: int = 6) -> str:
    return np.array2string(
        np.asarray(values, dtype=np.float64),
        precision=precision,
        suppress_small=True,
    )


def _as_bounds(
    bounds: np.ndarray | Iterable[Iterable[float]] | None,
) -> np.ndarray | None:
    if bounds is None:
        return None
    arr = np.asarray(bounds, dtype=np.float64)
    if arr.shape != (3, 2):
        raise ValueError(
            "workspace_bounds harus berbentuk (3, 2): [[xmin,xmax],[ymin,ymax],[zmin,zmax]]."
        )
    if np.any(arr[:, 0] > arr[:, 1]):
        raise ValueError(
            "workspace_bounds tidak valid: batas bawah lebih besar dari batas atas."
        )
    return arr


@dataclass(slots=True)
class IKResult:
    success: bool
    message: str
    joint_names: tuple[str, ...]
    q_rad: np.ndarray
    q_deg: np.ndarray
    final_ee_pos: np.ndarray
    final_ee_quat: np.ndarray
    target_pos: np.ndarray
    target_quat: np.ndarray | None
    position_error: np.ndarray
    rotation_error: np.ndarray
    iterations: int
    attempt: int

    @property
    def position_error_norm(self) -> float:
        return float(np.linalg.norm(self.position_error))

    @property
    def rotation_error_norm(self) -> float:
        return float(np.linalg.norm(self.rotation_error))

    @property
    def ok_for_control(self) -> bool:
        """True jika hasil boleh dikirim ke actuator walau statusnya BEST_EFFORT."""
        return bool(np.all(np.isfinite(self.q_rad)))


class MyCobotIK:
    """
    Damped least-squares IK untuk myCobot di MuJoCo.

    Catatan penting untuk RL:
    - Solver ini punya `self.data` sendiri untuk komputasi kinematik.
    - Jangan pakai solver untuk menulis langsung ke `env.data.qpos`.
    - Di environment, ambil `result.q_rad`, lalu kirim ke position actuator robot.
    """

    def __init__(
        self,
        xml_file: str | Path = DEFAULT_XML_PATH,
        ee_site_name: str = DEFAULT_EE_SITE_NAME,
        joint_names: tuple[str, ...] = ARM_JOINT_NAMES,
    ) -> None:
        if mujoco is None:  # pragma: no cover - depends on runtime env
            raise ModuleNotFoundError(
                "Package `mujoco` belum tersedia. Install dependency dari requirements.txt "
                "atau aktifkan environment project terlebih dahulu."
            ) from _MUJOCO_IMPORT_ERROR

        self.xml_file = Path(xml_file).expanduser().resolve()
        self.ee_site_name = str(ee_site_name)
        self.joint_names = tuple(joint_names)

        if not self.xml_file.exists():
            raise FileNotFoundError(f"File XML MuJoCo tidak ditemukan: {self.xml_file}")

        self.model = mujoco.MjModel.from_xml_path(str(self.xml_file))
        self.data = mujoco.MjData(self.model)

        self.ee_site_id = int(
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, self.ee_site_name)
        )
        if self.ee_site_id < 0:
            raise ValueError(
                f"Site end-effector `{self.ee_site_name}` tidak ditemukan."
            )

        self.joint_ids: list[int] = []
        self.qpos_indices: list[int] = []
        self.dof_indices: list[int] = []
        lower_limits: list[float] = []
        upper_limits: list[float] = []

        for joint_name in self.joint_names:
            joint_id = int(
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            )
            if joint_id < 0:
                raise ValueError(f"Joint `{joint_name}` tidak ditemukan di model.")

            if self.model.jnt_limited[joint_id] == 0:
                raise ValueError(
                    f"Joint `{joint_name}` tidak memiliki limit. "
                    "Untuk IK yang aman di RL, tambahkan range pada joint di XML."
                )

            self.joint_ids.append(joint_id)
            self.qpos_indices.append(int(self.model.jnt_qposadr[joint_id]))
            self.dof_indices.append(int(self.model.jnt_dofadr[joint_id]))
            lower_limits.append(float(self.model.jnt_range[joint_id][0]))
            upper_limits.append(float(self.model.jnt_range[joint_id][1]))

        self.lower_limits_rad = np.asarray(lower_limits, dtype=np.float64)
        self.upper_limits_rad = np.asarray(upper_limits, dtype=np.float64)
        self._base_qpos = self.model.qpos0.copy()
        self._base_qvel = np.zeros(self.model.nv, dtype=np.float64)
        self._set_arm_configuration(self.home_configuration())

    @property
    def n_arm_joints(self) -> int:
        return len(self.joint_names)

    def home_configuration(self) -> np.ndarray:
        home = np.zeros(self.n_arm_joints, dtype=np.float64)
        for idx, qpos_index in enumerate(self.qpos_indices):
            home[idx] = float(self.model.qpos0[qpos_index])
        return home

    def has_site(self, site_name: str) -> bool:
        site_id = int(
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        )
        return site_id >= 0

    def has_joint(self, joint_name: str) -> bool:
        joint_id = int(
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        )
        return joint_id >= 0

    def extract_arm_qpos(self, qpos: np.ndarray) -> np.ndarray:
        qpos = np.asarray(qpos, dtype=np.float64).reshape(-1)
        if qpos.shape[0] != self.model.nq:
            raise ValueError(
                f"qpos harus panjang {self.model.nq}, tetapi menerima {qpos.shape[0]}."
            )
        return qpos[self.qpos_indices].copy()

    def sync_base_qpos(self, qpos: np.ndarray, qvel: np.ndarray | None = None) -> None:
        """
        Sinkronkan qpos solver dengan state eksternal tanpa mengubah environment.

        Ini berguna jika XML IK adalah scene penuh yang punya gripper/object/freejoint.
        Solver akan mempertahankan qpos non-arm dari state eksternal saat mencari IK.
        """
        qpos = np.asarray(qpos, dtype=np.float64).reshape(-1)
        if qpos.shape[0] != self.model.nq:
            raise ValueError(
                f"qpos harus panjang {self.model.nq}, tetapi menerima {qpos.shape[0]}."
            )
        self._base_qpos = qpos.copy()

        if qvel is not None:
            qvel = np.asarray(qvel, dtype=np.float64).reshape(-1)
            if qvel.shape[0] != self.model.nv:
                raise ValueError(
                    f"qvel harus panjang {self.model.nv}, tetapi menerima {qvel.shape[0]}."
                )
            self._base_qvel = qvel.copy()

    def sync_base_from_data(self, data: object) -> None:
        """Shortcut untuk `sync_base_qpos(env.data.qpos, env.data.qvel)`."""
        self.sync_base_qpos(np.asarray(data.qpos), np.asarray(data.qvel))

    def _clip_to_limits(self, q_rad: np.ndarray) -> np.ndarray:
        q_rad = np.asarray(q_rad, dtype=np.float64).reshape(-1)
        if q_rad.shape != (self.n_arm_joints,):
            raise ValueError(
                f"q_rad harus berisi {self.n_arm_joints} joint, tetapi menerima shape {q_rad.shape}."
            )
        return np.clip(q_rad, self.lower_limits_rad, self.upper_limits_rad)

    def clip_target_pos(
        self,
        target_pos: np.ndarray,
        workspace_bounds: np.ndarray | Iterable[Iterable[float]] | None,
    ) -> np.ndarray:
        target_pos = np.asarray(target_pos, dtype=np.float64).reshape(3)
        bounds = _as_bounds(workspace_bounds)
        if bounds is None:
            return target_pos
        return np.clip(target_pos, bounds[:, 0], bounds[:, 1])

    def _set_arm_configuration(self, q_rad: np.ndarray) -> None:
        q_rad = self._clip_to_limits(q_rad)
        self.data.qpos[:] = self._base_qpos
        self.data.qvel[:] = self._base_qvel
        for qpos_index, value in zip(self.qpos_indices, q_rad):
            self.data.qpos[qpos_index] = float(value)
        mujoco.mj_forward(self.model, self.data)

    def site_pose(self, site_name: str) -> tuple[np.ndarray, np.ndarray]:
        site_id = int(
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        )
        if site_id < 0:
            raise ValueError(f"Site `{site_name}` tidak ditemukan di model.")
        quat = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quat, self.data.site_xmat[site_id])
        return self.data.site_xpos[site_id].copy(), _normalize_quat(quat)

    def end_effector_pose(
        self, q_rad: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        if q_rad is not None:
            self._set_arm_configuration(q_rad)
        return self.site_pose(self.ee_site_name)

    def _arm_jacobian(self) -> tuple[np.ndarray, np.ndarray]:
        jacp_full = np.zeros((3, self.model.nv), dtype=np.float64)
        jacr_full = np.zeros((3, self.model.nv), dtype=np.float64)
        mujoco.mj_jacSite(self.model, self.data, jacp_full, jacr_full, self.ee_site_id)
        return jacp_full[:, self.dof_indices], jacr_full[:, self.dof_indices]

    def solve(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray | None = None,
        *,
        initial_q: np.ndarray | None = None,
        position_only: bool = False,
        max_iters: int = 200,
        position_tolerance: float = 1e-3,
        rotation_tolerance: float = np.deg2rad(2.0),
        damping: float = 1e-3,
        step_size: float = 0.75,
        max_delta: float = np.deg2rad(12.0),
        rotation_weight: float = 0.35,
        random_restarts: int = 6,
        seed: int | None = 0,
    ) -> IKResult:
        target_pos = np.asarray(target_pos, dtype=np.float64).reshape(3)
        use_rotation = (target_quat is not None) and not position_only
        if use_rotation:
            target_quat = _normalize_quat(
                np.asarray(target_quat, dtype=np.float64).reshape(4)
            )
        else:
            target_quat = None

        if initial_q is None:
            initial_q = self.home_configuration()
        initial_q = self._clip_to_limits(
            np.asarray(initial_q, dtype=np.float64).reshape(-1)
        )

        rng = np.random.default_rng(seed)
        seeds = [initial_q]
        for _ in range(max(0, int(random_restarts))):
            seeds.append(rng.uniform(self.lower_limits_rad, self.upper_limits_rad))

        best_result: IKResult | None = None
        best_score = float("inf")
        max_iters = max(1, int(max_iters))
        damping = max(float(damping), 1e-9)
        step_size = float(step_size)
        max_delta = float(max_delta)
        rotation_weight = float(rotation_weight)

        for attempt, seed_q in enumerate(seeds):
            q_rad = self._clip_to_limits(seed_q.copy())
            stagnation_steps = 0
            prev_error_norm = float("inf")
            last_iteration = 0

            for iteration in range(1, max_iters + 1):
                last_iteration = iteration
                self._set_arm_configuration(q_rad)
                ee_pos, ee_quat = self.end_effector_pose()
                pos_error = target_pos - ee_pos
                rot_error = (
                    _rotation_vector(ee_quat, target_quat)
                    if target_quat is not None
                    else np.zeros(3, dtype=np.float64)
                )

                pos_norm = float(np.linalg.norm(pos_error))
                rot_norm = float(np.linalg.norm(rot_error))
                if pos_norm <= position_tolerance and (
                    target_quat is None or rot_norm <= rotation_tolerance
                ):
                    return IKResult(
                        success=True,
                        message="IK converged.",
                        joint_names=self.joint_names,
                        q_rad=q_rad.copy(),
                        q_deg=np.rad2deg(q_rad.copy()),
                        final_ee_pos=ee_pos,
                        final_ee_quat=ee_quat,
                        target_pos=target_pos.copy(),
                        target_quat=None if target_quat is None else target_quat.copy(),
                        position_error=pos_error,
                        rotation_error=rot_error,
                        iterations=iteration,
                        attempt=attempt,
                    )

                jacp, jacr = self._arm_jacobian()
                if target_quat is None:
                    jacobian = jacp
                    error = pos_error
                else:
                    jacobian = np.vstack([jacp, rotation_weight * jacr])
                    error = np.concatenate([pos_error, rotation_weight * rot_error])

                damp_matrix = (damping**2) * np.eye(jacobian.shape[0], dtype=np.float64)
                try:
                    delta_q = jacobian.T @ np.linalg.solve(
                        jacobian @ jacobian.T + damp_matrix,
                        error,
                    )
                except np.linalg.LinAlgError:
                    delta_q = np.linalg.pinv(jacobian) @ error

                delta_q = np.asarray(delta_q, dtype=np.float64)
                if max_delta > 0.0:
                    delta_q = np.clip(delta_q, -max_delta, max_delta)

                q_rad = self._clip_to_limits(q_rad + step_size * delta_q)

                error_norm = float(np.linalg.norm(error))
                if abs(prev_error_norm - error_norm) < 1e-10:
                    stagnation_steps += 1
                else:
                    stagnation_steps = 0
                prev_error_norm = error_norm
                if stagnation_steps >= 10:
                    break

            self._set_arm_configuration(q_rad)
            ee_pos, ee_quat = self.end_effector_pose()
            pos_error = target_pos - ee_pos
            rot_error = (
                _rotation_vector(ee_quat, target_quat)
                if target_quat is not None
                else np.zeros(3, dtype=np.float64)
            )

            score = float(
                np.linalg.norm(pos_error)
                + (
                    0.0
                    if target_quat is None
                    else rotation_weight * np.linalg.norm(rot_error)
                )
            )
            if score < best_score:
                best_score = score
                best_result = IKResult(
                    success=False,
                    message="IK belum memenuhi toleransi, mengembalikan solusi terbaik.",
                    joint_names=self.joint_names,
                    q_rad=q_rad.copy(),
                    q_deg=np.rad2deg(q_rad.copy()),
                    final_ee_pos=ee_pos,
                    final_ee_quat=ee_quat,
                    target_pos=target_pos.copy(),
                    target_quat=None if target_quat is None else target_quat.copy(),
                    position_error=pos_error,
                    rotation_error=rot_error,
                    iterations=last_iteration,
                    attempt=attempt,
                )

        if best_result is None:
            raise RuntimeError("IK gagal dijalankan karena tidak ada kandidat solusi.")
        return best_result

    def solve_from_current(
        self,
        current_q: np.ndarray,
        target_pos: np.ndarray,
        target_quat: np.ndarray | None = None,
        *,
        position_only: bool = True,
        config: IKRuntimeConfig = RL_FAST_CONFIG,
    ) -> IKResult:
        """
        Solve IK dari konfigurasi joint saat ini.

        Method ini disiapkan untuk `env.step()` RL: ringan, tanpa random restart,
        dan menggunakan toleransi yang cukup realistis untuk kontrol incremental.
        """
        return self.solve(
            target_pos=target_pos,
            target_quat=target_quat,
            initial_q=current_q,
            position_only=position_only,
            max_iters=config.max_iters,
            position_tolerance=config.position_tolerance,
            rotation_tolerance=config.rotation_tolerance,
            damping=config.damping,
            step_size=config.step_size,
            max_delta=config.max_delta,
            rotation_weight=config.rotation_weight,
            random_restarts=config.random_restarts,
            seed=config.seed,
        )

    def solve_delta(
        self,
        delta_pos: np.ndarray,
        *,
        current_q: np.ndarray | None = None,
        current_full_qpos: np.ndarray | None = None,
        current_full_qvel: np.ndarray | None = None,
        target_quat: np.ndarray | None = None,
        position_only: bool = True,
        workspace_bounds: np.ndarray | Iterable[Iterable[float]] | None = None,
        config: IKRuntimeConfig = RL_FAST_CONFIG,
    ) -> IKResult:
        """
        Kontrol incremental end-effector untuk SAC.

        Parameters
        ----------
        delta_pos:
            Perpindahan target EE `[dx, dy, dz]` dalam meter.
        current_q:
            6 joint arm saat ini. Jika None, akan diambil dari `current_full_qpos`.
        current_full_qpos/current_full_qvel:
            State MuJoCo environment. Tidak dimodifikasi, hanya disalin ke internal IK solver.
        workspace_bounds:
            Optional batas workspace `[[xmin,xmax],[ymin,ymax],[zmin,zmax]]`.
        """
        if current_full_qpos is not None:
            self.sync_base_qpos(current_full_qpos, current_full_qvel)
            if current_q is None:
                current_q = self.extract_arm_qpos(
                    np.asarray(current_full_qpos, dtype=np.float64)
                )

        if current_q is None:
            current_q = self.home_configuration()
        current_q = self._clip_to_limits(current_q)

        self._set_arm_configuration(current_q)
        current_ee_pos, current_ee_quat = self.end_effector_pose()
        delta_pos = np.asarray(delta_pos, dtype=np.float64).reshape(3)
        target_pos = self.clip_target_pos(current_ee_pos + delta_pos, workspace_bounds)

        if target_quat is None and not position_only:
            target_quat = current_ee_quat

        return self.solve_from_current(
            current_q=current_q,
            target_pos=target_pos,
            target_quat=target_quat,
            position_only=position_only,
            config=config,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inverse kinematics numerik untuk robot myCobot di MuJoCo. "
            "Solver bekerja pada site `attachment_site` dan 6 joint arm utama."
        )
    )
    parser.add_argument(
        "--xml-file",
        default=str(DEFAULT_XML_PATH),
        help="Path ke XML MuJoCo. Default: robot.xml di folder script atau source/robot/robot.xml.",
    )
    parser.add_argument(
        "--ee-site-name",
        default=DEFAULT_EE_SITE_NAME,
        help="Nama site end-effector yang ingin dikendalikan.",
    )
    parser.add_argument(
        "--target-pos",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help="Target posisi end-effector dalam meter.",
    )
    parser.add_argument(
        "--target-quat",
        nargs=4,
        type=float,
        metavar=("W", "X", "Y", "Z"),
        help="Target orientasi quaternion format w x y z.",
    )
    parser.add_argument(
        "--target-rpy-deg",
        nargs=3,
        type=float,
        metavar=("ROLL", "PITCH", "YAW"),
        help="Target orientasi Euler XYZ dalam derajat.",
    )
    parser.add_argument(
        "--target-site-name",
        default=None,
        help="Ambil target pose langsung dari site lain di XML.",
    )
    parser.add_argument(
        "--initial-joints-rad",
        nargs=6,
        type=float,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
        help="Seed awal joint dalam radian.",
    )
    parser.add_argument(
        "--initial-joints-deg",
        nargs=6,
        type=float,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
        help="Seed awal joint dalam derajat.",
    )
    parser.add_argument(
        "--position-only",
        action="store_true",
        help="Abaikan orientasi dan hanya cari solusi posisi.",
    )
    parser.add_argument("--max-iters", type=int, default=250)
    parser.add_argument("--position-tol-mm", type=float, default=1.0)
    parser.add_argument("--rotation-tol-deg", type=float, default=2.0)
    parser.add_argument("--damping", type=float, default=1e-3)
    parser.add_argument("--step-size", type=float, default=0.75)
    parser.add_argument("--max-delta-deg", type=float, default=12.0)
    parser.add_argument("--rotation-weight", type=float, default=0.35)
    parser.add_argument("--random-restarts", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--render",
        action="store_true",
        help="Buka MuJoCo viewer untuk melihat pose hasil IK.",
    )
    parser.add_argument(
        "--animate-seconds",
        type=float,
        default=2.0,
        help="Durasi animasi dari seed awal ke solusi IK.",
    )
    parser.add_argument(
        "--render-seconds",
        type=float,
        default=0.0,
        help="Lama viewer ditahan dalam detik. 0 berarti sampai jendela ditutup.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=60.0,
        help="FPS update viewer saat animasi/render.",
    )
    return parser.parse_args()


def _resolve_initial_joints(args: argparse.Namespace) -> np.ndarray | None:
    if args.initial_joints_rad is not None and args.initial_joints_deg is not None:
        raise ValueError(
            "Pilih salah satu: --initial-joints-rad atau --initial-joints-deg."
        )
    if args.initial_joints_rad is not None:
        return np.asarray(args.initial_joints_rad, dtype=np.float64)
    if args.initial_joints_deg is not None:
        return np.deg2rad(np.asarray(args.initial_joints_deg, dtype=np.float64))
    return None


def _resolve_target_pose(
    solver: MyCobotIK, args: argparse.Namespace
) -> tuple[np.ndarray, np.ndarray | None]:
    if args.target_site_name:
        target_pos, target_quat = solver.site_pose(args.target_site_name)
        if args.position_only:
            return target_pos, None
        return target_pos, target_quat

    if args.target_pos is None:
        raise ValueError(
            "Target belum ditentukan. Gunakan --target-pos X Y Z atau --target-site-name NAMA_SITE."
        )

    target_pos = np.asarray(args.target_pos, dtype=np.float64)
    if args.position_only:
        return target_pos, None

    if args.target_quat is not None and args.target_rpy_deg is not None:
        raise ValueError("Pilih salah satu: --target-quat atau --target-rpy-deg.")
    if args.target_quat is not None:
        return target_pos, np.asarray(args.target_quat, dtype=np.float64)
    if args.target_rpy_deg is not None:
        roll, pitch, yaw = np.deg2rad(np.asarray(args.target_rpy_deg, dtype=np.float64))
        return target_pos, _quat_from_euler_xyz(float(roll), float(pitch), float(yaw))
    return target_pos, None


def render_solution(
    solver: MyCobotIK,
    result: IKResult,
    *,
    initial_q: np.ndarray | None = None,
    animate_seconds: float = 2.0,
    render_seconds: float = 0.0,
    fps: float = 60.0,
) -> None:
    try:
        import mujoco.viewer
    except (
        ModuleNotFoundError
    ) as exc:  # pragma: no cover - viewer depends on runtime env
        raise ModuleNotFoundError(
            "mujoco.viewer tidak tersedia di environment ini. "
            "Pastikan package `mujoco` sudah terpasang dengan dukungan viewer."
        ) from exc

    start_q = (
        solver.home_configuration()
        if initial_q is None
        else np.asarray(initial_q, dtype=np.float64).reshape(-1)
    )
    start_q = solver._clip_to_limits(start_q)
    end_q = solver._clip_to_limits(result.q_rad)
    fps = max(float(fps), 1.0)
    animate_seconds = max(float(animate_seconds), 0.0)
    render_seconds = float(render_seconds)

    def sync_target_marker() -> None:
        viewer.user_scn.ngeom = 0
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[0],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=np.array([0.01, 0.0, 0.0], dtype=np.float64),
            pos=np.asarray(result.target_pos, dtype=np.float64),
            mat=np.eye(3, dtype=np.float64).reshape(-1),
            rgba=np.array([1.0, 0.9, 0.1, 0.85], dtype=np.float32),
        )
        ngeom = 1

        if result.target_quat is not None:
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[ngeom],
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=np.array([0.018, 0.004, 0.004], dtype=np.float64),
                pos=np.asarray(result.target_pos, dtype=np.float64),
                mat=_quat_to_mat(result.target_quat).reshape(-1),
                rgba=np.array([1.0, 0.45, 0.15, 0.35], dtype=np.float32),
            )
            ngeom += 1

        viewer.user_scn.ngeom = ngeom

    solver._set_arm_configuration(start_q)
    with mujoco.viewer.launch_passive(solver.model, solver.data) as viewer:
        if animate_seconds > 0.0:
            total_frames = max(1, int(np.ceil(animate_seconds * fps)))
            for frame_idx in range(total_frames + 1):
                if not viewer.is_running():
                    return
                alpha = frame_idx / total_frames
                q_frame = (1.0 - alpha) * start_q + alpha * end_q
                with viewer.lock():
                    solver._set_arm_configuration(q_frame)
                    sync_target_marker()
                viewer.sync()
                time.sleep(1.0 / fps)

        with viewer.lock():
            solver._set_arm_configuration(end_q)
            sync_target_marker()
        if render_seconds <= 0.0:
            while viewer.is_running():
                with viewer.lock():
                    sync_target_marker()
                viewer.sync()
                time.sleep(1.0 / fps)
            return

        end_time = time.time() + render_seconds
        while viewer.is_running() and time.time() < end_time:
            with viewer.lock():
                sync_target_marker()
            viewer.sync()
            time.sleep(1.0 / fps)


def main() -> None:
    args = parse_args()
    solver = MyCobotIK(xml_file=args.xml_file, ee_site_name=args.ee_site_name)
    target_pos, target_quat = _resolve_target_pose(solver, args)
    initial_q = _resolve_initial_joints(args)

    result = solver.solve(
        target_pos=target_pos,
        target_quat=target_quat,
        initial_q=initial_q,
        position_only=bool(args.position_only),
        max_iters=int(args.max_iters),
        position_tolerance=float(args.position_tol_mm) / 1000.0,
        rotation_tolerance=np.deg2rad(float(args.rotation_tol_deg)),
        damping=float(args.damping),
        step_size=float(args.step_size),
        max_delta=np.deg2rad(float(args.max_delta_deg)),
        rotation_weight=float(args.rotation_weight),
        random_restarts=int(args.random_restarts),
        seed=int(args.seed),
    )

    final_rpy_deg = np.rad2deg(_quat_to_euler_xyz(result.final_ee_quat))

    print(f"Status          : {'SUCCESS' if result.success else 'BEST_EFFORT'}")
    print(f"Message         : {result.message}")
    print(f"Attempt         : {result.attempt}")
    print(f"Iterations      : {result.iterations}")
    print(f"Joint names     : {', '.join(result.joint_names)}")
    print(f"Joint rad       : {_format_values(result.q_rad, precision=6)}")
    print(f"Joint deg       : {_format_values(result.q_deg, precision=3)}")
    print(f"EE pos final    : {_format_values(result.final_ee_pos, precision=6)} m")
    print(f"EE quat final   : {_format_values(result.final_ee_quat, precision=6)}")
    print(f"EE rpy final    : {_format_values(final_rpy_deg, precision=3)} deg")
    print(f"Target pos      : {_format_values(result.target_pos, precision=6)} m")
    if result.target_quat is not None:
        target_rpy_deg = np.rad2deg(_quat_to_euler_xyz(result.target_quat))
        print(f"Target quat     : {_format_values(result.target_quat, precision=6)}")
        print(f"Target rpy      : {_format_values(target_rpy_deg, precision=3)} deg")
    print(
        f"Pos error       : {_format_values(result.position_error, precision=6)} m "
        f"(norm={result.position_error_norm * 1000.0:.3f} mm)"
    )
    print(
        f"Rot error       : {_format_values(result.rotation_error, precision=6)} rad "
        f"(norm={np.rad2deg(result.rotation_error_norm):.3f} deg)"
    )

    if args.render:
        render_solution(
            solver,
            result,
            initial_q=initial_q,
            animate_seconds=float(args.animate_seconds),
            render_seconds=float(args.render_seconds),
            fps=float(args.fps),
        )

    if not result.success:
        raise SystemExit(1)


__all__ = [
    "ARM_JOINT_NAMES",
    "DEFAULT_EE_SITE_NAME",
    "DEFAULT_XML_PATH",
    "IKResult",
    "IKRuntimeConfig",
    "MyCobotIK",
    "RL_FAST_CONFIG",
]


if __name__ == "__main__":
    main()
