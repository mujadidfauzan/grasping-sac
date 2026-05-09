from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Callable
import imageio
import gymnasium as gym
import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

# ============================================================
# Path setup
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from source.envs import GraspingEnvIK  # noqa: E402

# ============================================================
# Custom callback untuk logging info reward dari environment
# ============================================================


class InfoLoggingCallback(BaseCallback):
    """
    Callback ini membaca dictionary `info` dari env.step(),
    lalu mencatat komponen reward ke TensorBoard.

    Ini penting untuk riset supaya kamu bisa lihat:
    - reward_distance
    - reward_near_bonus
    - reward_lift_dense
    - reward_lift_bonus
    - reward_action_penalty
    - distance_ee_object
    - lift_amount
    - success_counter
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])

        if len(infos) == 0:
            return True

        # Karena kita pakai DummyVecEnv dengan 1 env,
        # biasanya infos hanya berisi satu dict.
        info = infos[0]

        keys_to_log = [
            "reward_total",
            "reward_distance",
            "reward_near_bonus",
            "reward_lift_dense",
            "reward_lift_bonus",
            "reward_action_penalty",
            "distance_ee_object",
            "lift_amount",
            "object_z",
            "action_norm_sq",
            "ik_error_norm",
            "success_counter",
        ]

        for key in keys_to_log:
            if key in info:
                value = info[key]
                if isinstance(value, (int, float, np.integer, np.floating, bool)):
                    self.logger.record(f"env/{key}", float(value))

        if "is_success" in info:
            self.logger.record("env/is_success", float(info["is_success"]))

        if "ik_success" in info:
            self.logger.record("env/ik_success", float(info["ik_success"]))

        if "ik_failed" in info:
            self.logger.record("env/ik_failed", float(info["ik_failed"]))

        return True


class VideoRecordCallback(BaseCallback):
    def __init__(
        self,
        xml_path,
        video_dir="videos",
        record_freq=25000,
        video_length=300,
        verbose=0,
    ):
        super().__init__(verbose)
        self.xml_path = xml_path
        self.video_dir = Path(video_dir)
        self.record_freq = int(record_freq)
        self.video_length = int(video_length)
        self.video_dir.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.record_freq != 0:
            return True

        env = GraspingEnvIK(
            xml_path=self.xml_path,
            render_mode="rgb_array",
            max_episode_steps=self.video_length,
            randomize_object=False,
        )

        obs, info = env.reset()
        frames = []

        for _ in range(self.video_length):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            frame = env.render()
            frames.append(frame)

            if terminated or truncated:
                break

        video_path = self.video_dir / f"record_step_{self.num_timesteps}.mp4"
        imageio.mimsave(video_path, frames, fps=30)

        env.close()

        if self.verbose:
            print(f"[VIDEO] Saved: {video_path}")

        return True


# ============================================================
# Environment factory
# ============================================================


def make_env(
    xml_path: str | Path,
    render_mode: str | None = None,
    max_episode_steps: int = 150,
    frame_skip: int = 10,
    action_scale: float = 0.02,
    close_distance_threshold: float = 0.04,
    lift_height: float = 0.05,
    randomize_object: bool = True,
) -> Callable[[], gym.Env]:
    """
    Factory function untuk membuat environment.

    SB3 biasanya menerima VecEnv.
    Karena itu env dibuat dalam function agar bisa dibungkus DummyVecEnv.
    """

    def _init() -> gym.Env:
        env = GraspingEnvIK(
            xml_path=xml_path,
            render_mode=render_mode,
            max_episode_steps=max_episode_steps,
            frame_skip=frame_skip,
            action_scale=action_scale,
            close_distance_threshold=close_distance_threshold,
            lift_height=lift_height,
            randomize_object=randomize_object,
        )

        env = Monitor(env)
        return env

    return _init


# ============================================================
# Main training
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SAC untuk myCobot grasping di MuJoCo."
    )

    parser.add_argument(
        "--xml",
        type=str,
        default="object_lift_grasp.xml",
        help="Path ke XML MuJoCo scene.",
    )

    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=200_000,
        help="Jumlah total timestep training.",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Learning rate SAC.",
    )

    parser.add_argument(
        "--buffer-size",
        type=int,
        default=300_000,
        help="Ukuran replay buffer.",
    )

    parser.add_argument(
        "--learning-starts",
        type=int,
        default=5_000,
        help="Jumlah step random sebelum SAC mulai update network.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size untuk update SAC.",
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.98,
        help="Discount factor.",
    )

    parser.add_argument(
        "--tau",
        type=float,
        default=0.02,
        help="Soft update coefficient untuk target network.",
    )

    parser.add_argument(
        "--train-freq",
        type=int,
        default=1,
        help="Update model setiap N environment step.",
    )

    parser.add_argument(
        "--gradient-steps",
        type=int,
        default=1,
        help="Jumlah gradient step setiap update.",
    )

    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=150,
        help="Maksimum step per episode.",
    )

    parser.add_argument(
        "--frame-skip",
        type=int,
        default=10,
        help="Jumlah MuJoCo step untuk setiap 1 action RL.",
    )

    parser.add_argument(
        "--action-scale",
        type=float,
        default=0.02,
        help="Maksimum delta posisi EE per step. 0.02 berarti 2 cm.",
    )

    parser.add_argument(
        "--close-distance-threshold",
        type=float,
        default=0.04,
        help="Threshold EE dianggap dekat dengan objek. 0.04 berarti 4 cm.",
    )

    parser.add_argument(
        "--lift-height",
        type=float,
        default=0.05,
        help="Tinggi minimal objek dianggap berhasil terangkat. 0.05 berarti 5 cm.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    parser.add_argument(
        "--log-dir",
        type=str,
        default="runs/sac_mycobot_grasp",
        help="Folder log TensorBoard.",
    )

    parser.add_argument(
        "--save-dir",
        type=str,
        default="models",
        help="Folder penyimpanan model.",
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default="sac_distance_lift_v1",
        help="Nama run TensorBoard dan model.",
    )

    parser.add_argument(
        "--check-env",
        action="store_true",
        help="Jalankan check_env sebelum training.",
    )

    parser.add_argument(
        "--no-randomize-object",
        action="store_true",
        help="Matikan randomisasi posisi objek.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device training.",
    )

    return parser.parse_args()


def record_check_env_episode(
    xml_path: str | Path,
    video_path: str | Path = "videos/check_env_record.mp4",
    steps: int = 300,
    max_episode_steps: int = 150,
    frame_skip: int = 10,
    action_scale: float = 0.02,
    close_distance_threshold: float = 0.04,
    lift_height: float = 0.05,
    randomize_object: bool = True,
    seed: int = 42,
) -> None:
    video_path = Path(video_path).expanduser().resolve()
    video_path.parent.mkdir(parents=True, exist_ok=True)

    env = GraspingEnvIK(
        xml_path=xml_path,
        render_mode="rgb_array",
        max_episode_steps=max_episode_steps,
        frame_skip=frame_skip,
        action_scale=action_scale,
        close_distance_threshold=close_distance_threshold,
        lift_height=lift_height,
        randomize_object=randomize_object,
    )

    obs, info = env.reset(seed=seed)
    frames = []

    for step_idx in range(int(steps)):
        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)

        frame = env.render()
        if frame is not None:
            frames.append(frame)

        print(
            f"[CHECK_RECORD] step={step_idx + 1:04d} "
            f"reward={reward:.4f} "
            f"distance={info.get('distance_ee_object', -1):.4f} "
            f"lift={info.get('lift_amount', -1):.4f} "
            f"success={info.get('is_success', False)}"
        )

        if terminated or truncated:
            obs, info = env.reset(seed=seed + step_idx + 1)

    env.close()

    if len(frames) == 0:
        raise RuntimeError("Tidak ada frame yang berhasil direkam dari env.render().")

    imageio.mimsave(video_path, frames, fps=30)
    print(f"[INFO] Video check_env disimpan ke: {video_path}")


def main() -> None:
    args = parse_args()

    xml_path = Path(args.xml).expanduser().resolve()
    if not xml_path.exists():
        raise FileNotFoundError(
            f"XML file tidak ditemukan: {xml_path}\n"
            f"Pastikan file XML berada di folder project atau berikan path dengan --xml."
        )

    log_dir = Path(args.log_dir).expanduser().resolve()
    save_dir = Path(args.save_dir).expanduser().resolve()

    log_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)

    model_save_path = save_dir / f"{args.run_name}_final"
    best_model_dir = save_dir / f"{args.run_name}_best"
    checkpoint_dir = save_dir / f"{args.run_name}_checkpoints"

    best_model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    randomize_object = not args.no_randomize_object

    # ------------------------------------------------------------
    # Optional environment check
    # ------------------------------------------------------------
    if args.check_env:
        print("[INFO] Running SB3 check_env...")

        raw_env = GraspingEnvIK(
            xml_path=xml_path,
            render_mode=None,
            max_episode_steps=args.max_episode_steps,
            frame_skip=args.frame_skip,
            action_scale=args.action_scale,
            close_distance_threshold=args.close_distance_threshold,
            lift_height=args.lift_height,
            randomize_object=randomize_object,
        )

        check_env(raw_env, warn=True, skip_render_check=True)
        raw_env.close()

        print("[INFO] check_env selesai.")
        print("[INFO] Recording check_env random episode...")

        record_check_env_episode(
            xml_path=xml_path,
            video_path=save_dir / f"{args.run_name}_check_env_record.mp4",
            steps=args.max_episode_steps,
            max_episode_steps=args.max_episode_steps,
            frame_skip=args.frame_skip,
            action_scale=args.action_scale,
            close_distance_threshold=args.close_distance_threshold,
            lift_height=args.lift_height,
            randomize_object=randomize_object,
            seed=args.seed,
        )
    # ------------------------------------------------------------
    # Training environment
    # ------------------------------------------------------------

    env = DummyVecEnv(
        [
            make_env(
                xml_path=xml_path,
                render_mode=None,
                max_episode_steps=args.max_episode_steps,
                frame_skip=args.frame_skip,
                action_scale=args.action_scale,
                close_distance_threshold=args.close_distance_threshold,
                lift_height=args.lift_height,
                randomize_object=randomize_object,
            )
        ]
    )
    env = VecMonitor(env)

    # ------------------------------------------------------------
    # Evaluation environment
    # ------------------------------------------------------------

    eval_env = DummyVecEnv(
        [
            make_env(
                xml_path=xml_path,
                render_mode=None,
                max_episode_steps=args.max_episode_steps,
                frame_skip=args.frame_skip,
                action_scale=args.action_scale,
                close_distance_threshold=args.close_distance_threshold,
                lift_height=args.lift_height,
                randomize_object=randomize_object,
            )
        ]
    )
    eval_env = VecMonitor(eval_env)

    # ------------------------------------------------------------
    # SAC model
    # ------------------------------------------------------------

    policy_kwargs = dict(
        net_arch=[256, 256],
    )

    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        tau=args.tau,
        gamma=args.gamma,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        ent_coef="auto",
        target_update_interval=1,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=args.seed,
        tensorboard_log=str(log_dir),
        device=args.device,
    )

    # ------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------

    video_callback = VideoRecordCallback(
        xml_path=xml_path,
        video_dir=save_dir / f"{args.run_name}_videos",
        record_freq=25_000,
        video_length=300,
        verbose=1,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=25_000,
        save_path=str(checkpoint_dir),
        name_prefix=args.run_name,
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(best_model_dir),
        log_path=str(log_dir / "eval"),
        eval_freq=10_000,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    info_logging_callback = InfoLoggingCallback()

    callbacks = CallbackList(
        [checkpoint_callback, eval_callback, info_logging_callback, video_callback]
    )

    # ------------------------------------------------------------
    # Train
    # ------------------------------------------------------------

    print("[INFO] Mulai training SAC...")
    print(f"[INFO] XML                 : {xml_path}")
    print(f"[INFO] Total timesteps     : {args.total_timesteps}")
    print(f"[INFO] Log dir             : {log_dir}")
    print(f"[INFO] Save dir            : {save_dir}")
    print(f"[INFO] Randomize object    : {randomize_object}")
    print(f"[INFO] Action scale        : {args.action_scale}")
    print(f"[INFO] Close threshold     : {args.close_distance_threshold}")
    print(f"[INFO] Lift height         : {args.lift_height}")

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        tb_log_name=args.run_name,
        progress_bar=True,
    )

    # ------------------------------------------------------------
    # Save final model
    # ------------------------------------------------------------

    model.save(str(model_save_path))
    print(f"[INFO] Model final disimpan ke: {model_save_path}.zip")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
