from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# Sesuaikan ini dengan nama file env kamu.
# Dari error sebelumnya, env kamu ada di source/envs/grasping_env_ik.py
from source.envs.grasping_env_ik import GraspingEnvIK


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test checkpoint SAC untuk myCobot grasping."
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path ke checkpoint .zip SAC.",
    )

    parser.add_argument(
        "--xml",
        type=str,
        default="source/robot/object_lift.xml",
        help="Path ke XML MuJoCo.",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Jumlah episode test.",
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
        help="Frame skip env.",
    )

    parser.add_argument(
        "--action-scale",
        type=float,
        default=0.02,
        help="Scale delta posisi EE.",
    )

    parser.add_argument(
        "--rotation-action-scale-deg",
        type=float,
        default=10.0,
        help="Scale delta roll/pitch/yaw EE dalam derajat.",
    )

    parser.add_argument(
        "--close-distance-threshold",
        type=float,
        default=0.04,
        help="Threshold EE dekat objek.",
    )

    parser.add_argument(
        "--lift-height",
        type=float,
        default=0.05,
        help="Threshold objek dianggap terangkat.",
    )

    parser.add_argument(
        "--randomize-object",
        action="store_true",
        help="Aktifkan random posisi objek saat test.",
    )

    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Gunakan aksi stochastic. Default deterministic.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    xml_path = Path(args.xml).expanduser().resolve()

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint tidak ditemukan: {checkpoint_path}")

    if not xml_path.exists():
        raise FileNotFoundError(f"XML tidak ditemukan: {xml_path}")

    env = GraspingEnvIK(
        xml_path=xml_path,
        render_mode="human",
        max_episode_steps=args.max_episode_steps,
        frame_skip=args.frame_skip,
        action_scale=args.action_scale,
        rotation_action_scale=np.deg2rad(args.rotation_action_scale_deg),
        close_distance_threshold=args.close_distance_threshold,
        lift_height=args.lift_height,
        randomize_object=args.randomize_object,
    )

    model = SAC.load(str(checkpoint_path), env=env)

    deterministic = not args.stochastic

    print("[INFO] Testing checkpoint")
    print(f"[INFO] Checkpoint     : {checkpoint_path}")
    print(f"[INFO] XML            : {xml_path}")
    print(f"[INFO] Episodes       : {args.episodes}")
    print(f"[INFO] Deterministic  : {deterministic}")
    print(f"[INFO] Random object  : {args.randomize_object}")
    print(f"[INFO] Rotation scale : {args.rotation_action_scale_deg} deg")

    for ep in range(args.episodes):
        obs, info = env.reset(seed=ep)

        episode_reward = 0.0
        min_distance = float("inf")
        max_lift = 0.0
        success = False

        for step in range(args.max_episode_steps):
            action, _ = model.predict(
                obs,
                deterministic=deterministic,
            )

            obs, reward, terminated, truncated, info = env.step(action)
            env.render()

            episode_reward += float(reward)

            distance = float(info.get("distance_ee_object", np.nan))
            lift_amount = float(info.get("lift_amount", 0.0))

            if np.isfinite(distance):
                min_distance = min(min_distance, distance)

            max_lift = max(max_lift, lift_amount)

            if info.get("is_success", False):
                success = True

            print(
                f"[EP {ep + 1:02d}] "
                f"step={step + 1:03d} "
                f"reward={reward: .4f} "
                f"total={episode_reward: .4f} "
                f"dist={distance: .4f} "
                f"lift={lift_amount: .4f} "
                f"success={info.get('is_success', False)} "
                f"action={np.round(action, 3)}"
            )

            if terminated or truncated:
                break

        print(
            f"\n[RESULT EP {ep + 1}] "
            f"total_reward={episode_reward:.4f} "
            f"min_distance={min_distance:.4f} "
            f"max_lift={max_lift:.4f} "
            f"success={success}\n"
        )

    env.close()


if __name__ == "__main__":
    main()
