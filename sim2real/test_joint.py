from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sim2real.remote import MyCobotRemote

DEFAULT_CSV_DIR = PROJECT_ROOT / "logs_eval" / "GraspingEnv"
DEFAULT_ROBOT_IP = "10.244.4.108"
DEFAULT_PORT = 5005
DEFAULT_SPEED = 20
DEFAULT_TIMEOUT = 2.0
DEFAULT_ACK_TIMEOUT = 5.0
DEFAULT_SETTLE_TIMEOUT = 20.0
DEFAULT_TOLERANCE_DEG = 3.0
DEFAULT_PREVIEW_COUNT = 8
DEFAULT_MAX_JUMP_DEG = 360.0
DEFAULT_SLEEP_SECONDS = 0.05
DEFAULT_RESAMPLE_MODE = "interpolate"

ARM_JOINT_COLUMNS = [
    "robot_joint_pos_link2_to_link1",
    "robot_joint_pos_link3_to_link2",
    "robot_joint_pos_link4_to_link3",
    "robot_joint_pos_link5_to_link4",
    "robot_joint_pos_link6_to_link5",
    "robot_joint_pos_link6output_to_link6",
]
GRIPPER_STATE_COLUMN = "gripper_state"

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


@dataclass(frozen=True)
class TrajectoryPoint:
    csv_row_index: int
    episode: int
    step: int
    phase: str
    target_rad: np.ndarray
    target_deg: np.ndarray
    gripper_state: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preview or replay robot joint positions from an evaluation CSV to MyCobot. "
            "By default this only previews; pass --execute to move the robot."
        )
    )
    parser.add_argument(
        "--csv",
        default=None,
        help=(
            "Path to evaluation CSV. If omitted, picks the newest *_debug_state.csv "
            f"under {DEFAULT_CSV_DIR}."
        ),
    )
    parser.add_argument("--robot-ip", default=DEFAULT_ROBOT_IP)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--speed", type=int, default=DEFAULT_SPEED)
    parser.add_argument("--ack-timeout", type=float, default=DEFAULT_ACK_TIMEOUT)
    parser.add_argument("--settle-timeout", type=float, default=DEFAULT_SETTLE_TIMEOUT)
    parser.add_argument("--tolerance-deg", type=float, default=DEFAULT_TOLERANCE_DEG)
    parser.add_argument(
        "--phase",
        choices=["all", "reset", "step"],
        default="step",
        help="Filter CSV rows by phase.",
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=None,
        help="Only use rows from one episode.",
    )
    parser.add_argument(
        "--start-step",
        type=int,
        default=None,
        help="First env step to include after phase filtering.",
    )
    parser.add_argument(
        "--end-step",
        type=int,
        default=None,
        help="Last env step to include after phase filtering.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Take every Nth filtered row.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of filtered rows to keep.",
    )
    parser.add_argument(
        "--target-points",
        type=int,
        default=None,
        help=(
            "Compress each episode to this many trajectory points. "
            "Example: 300 -> 30 or 10."
        ),
    )
    parser.add_argument(
        "--resample-mode",
        choices=["interpolate", "mean"],
        default=DEFAULT_RESAMPLE_MODE,
        help=(
            "How to compress trajectory when --target-points is used. "
            "`interpolate` is smoother and recommended for robot playback."
        ),
    )
    parser.add_argument(
        "--preview-count",
        type=int,
        default=DEFAULT_PREVIEW_COUNT,
        help="How many target rows to print before execution.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=DEFAULT_SLEEP_SECONDS,
        help="Extra pause after each command.",
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait until each commanded joint target is reached.",
    )
    parser.add_argument(
        "--use-gripper",
        action="store_true",
        help="Use CSV gripper_state column to open/close gripper.",
    )
    parser.add_argument("--gripper-speed", type=int, default=50)
    parser.add_argument(
        "--max-jump-deg",
        type=float,
        default=DEFAULT_MAX_JUMP_DEG,
        help=(
            "Abort if current robot pose differs from the next CSV target by more than "
            "this amount on any arm joint."
        ),
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually send the trajectory to MyCobot.",
    )
    return parser.parse_args()


def resolve_csv_path(csv_arg: str | None) -> Path:
    if csv_arg:
        csv_path = Path(csv_arg).expanduser()
        if not csv_path.is_absolute():
            csv_path = (PROJECT_ROOT / csv_path).resolve()
    else:
        if not DEFAULT_CSV_DIR.exists():
            raise FileNotFoundError(
                f"Default CSV directory not found: {DEFAULT_CSV_DIR}"
            )
        candidates = sorted(
            DEFAULT_CSV_DIR.glob("*_debug_state.csv"),
            key=lambda path: path.stat().st_mtime,
        )
        if not candidates:
            raise FileNotFoundError(
                f"No *_debug_state.csv found under {DEFAULT_CSV_DIR}"
            )
        csv_path = candidates[-1]

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    return csv_path


def format_values(values: np.ndarray | list[float]) -> str:
    return ", ".join(f"{float(value):.3f}" for value in values)


def wrap_joint_error_deg(current_deg: np.ndarray, target_deg: np.ndarray) -> np.ndarray:
    return (current_deg - target_deg + 180.0) % 360.0 - 180.0


def clip_joint_targets_deg(target_deg: np.ndarray) -> np.ndarray:
    lower = JOINT_LIMITS_DEG[:, 0]
    upper = JOINT_LIMITS_DEG[:, 1]
    return np.clip(target_deg, lower, upper)


def parse_gripper_state(row: dict[str, str]) -> int | None:
    value = row.get(GRIPPER_STATE_COLUMN)
    if value is None:
        return None

    state = value.strip().lower()
    if state == "closed":
        return 1
    if state == "open":
        return 0
    return None


def parse_trajectory_point(row_index: int, row: dict[str, str]) -> TrajectoryPoint:
    try:
        episode = int(row["episode"])
        step = int(row["step"])
        phase = row["phase"].strip()
    except KeyError as exc:
        raise KeyError(f"Missing required CSV column: {exc}") from exc

    missing_joint_columns = [
        column for column in ARM_JOINT_COLUMNS if column not in row or row[column] == ""
    ]
    if missing_joint_columns:
        missing_text = ", ".join(missing_joint_columns)
        raise KeyError(
            "CSV does not contain all required robot joint columns: " f"{missing_text}"
        )

    target_rad = np.array(
        [float(row[column]) for column in ARM_JOINT_COLUMNS], dtype=np.float64
    )
    target_deg = clip_joint_targets_deg(np.rad2deg(target_rad))

    return TrajectoryPoint(
        csv_row_index=row_index,
        episode=episode,
        step=step,
        phase=phase,
        target_rad=target_rad,
        target_deg=target_deg,
        gripper_state=parse_gripper_state(row),
    )


def load_trajectory(
    csv_path: Path,
    *,
    episode: int | None,
    phase: str,
    start_step: int | None,
    end_step: int | None,
    stride: int,
    limit: int | None,
) -> list[TrajectoryPoint]:
    if stride < 1:
        raise ValueError("--stride must be at least 1.")
    if limit is not None and limit < 1:
        raise ValueError("--limit must be at least 1 when provided.")

    points: list[TrajectoryPoint] = []
    with csv_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row_index, row in enumerate(reader, start=2):
            point = parse_trajectory_point(row_index, row)

            if episode is not None and point.episode != episode:
                continue
            if phase != "all" and point.phase != phase:
                continue
            if start_step is not None and point.step < start_step:
                continue
            if end_step is not None and point.step > end_step:
                continue

            points.append(point)

    points = points[::stride]
    if limit is not None:
        points = points[:limit]

    if not points:
        raise ValueError("No CSV rows matched the provided filters.")

    return points


def build_point_from_deg(
    reference_point: TrajectoryPoint,
    *,
    step: int,
    target_deg: np.ndarray,
    gripper_state: int | None,
) -> TrajectoryPoint:
    clipped_target_deg = clip_joint_targets_deg(
        np.asarray(target_deg, dtype=np.float64)
    )
    target_rad = np.deg2rad(clipped_target_deg)
    return TrajectoryPoint(
        csv_row_index=reference_point.csv_row_index,
        episode=reference_point.episode,
        step=int(step),
        phase=reference_point.phase,
        target_rad=target_rad,
        target_deg=clipped_target_deg,
        gripper_state=gripper_state,
    )


def summarize_gripper_states(states: list[int | None]) -> int | None:
    valid_states = [state for state in states if state is not None]
    if not valid_states:
        return None

    closed_count = sum(1 for state in valid_states if state == 1)
    open_count = sum(1 for state in valid_states if state == 0)
    if closed_count == open_count:
        return valid_states[-1]
    return 1 if closed_count > open_count else 0


def resample_episode_interpolate(
    points: list[TrajectoryPoint], target_points: int
) -> list[TrajectoryPoint]:
    if target_points >= len(points):
        return points

    old_positions = np.arange(len(points), dtype=np.float64)
    new_positions = np.linspace(0.0, len(points) - 1, num=target_points)
    joint_matrix_deg = np.vstack([point.target_deg for point in points])
    step_values = np.array([point.step for point in points], dtype=np.float64)

    resampled_joint_matrix = np.vstack(
        [
            np.interp(new_positions, old_positions, joint_matrix_deg[:, joint_index])
            for joint_index in range(joint_matrix_deg.shape[1])
        ]
    ).T
    resampled_steps = np.rint(
        np.interp(new_positions, old_positions, step_values)
    ).astype(int)
    nearest_indices = np.rint(new_positions).astype(int)

    resampled_points: list[TrajectoryPoint] = []
    for index, nearest_index in enumerate(nearest_indices):
        reference_point = points[int(nearest_index)]
        resampled_points.append(
            build_point_from_deg(
                reference_point,
                step=int(resampled_steps[index]),
                target_deg=resampled_joint_matrix[index],
                gripper_state=reference_point.gripper_state,
            )
        )

    return resampled_points


def resample_episode_mean(
    points: list[TrajectoryPoint], target_points: int
) -> list[TrajectoryPoint]:
    if target_points >= len(points):
        return points

    boundaries = np.linspace(0, len(points), num=target_points + 1)
    resampled_points: list[TrajectoryPoint] = []

    for index in range(target_points):
        start = int(np.floor(boundaries[index]))
        end = int(np.floor(boundaries[index + 1]))
        end = max(end, start + 1)
        chunk = points[start:end]
        if not chunk:
            continue

        joint_matrix_deg = np.vstack([point.target_deg for point in chunk])
        mean_target_deg = np.mean(joint_matrix_deg, axis=0)
        mean_step = int(round(np.mean([point.step for point in chunk])))
        reference_point = chunk[len(chunk) // 2]
        gripper_state = summarize_gripper_states(
            [point.gripper_state for point in chunk]
        )
        resampled_points.append(
            build_point_from_deg(
                reference_point,
                step=mean_step,
                target_deg=mean_target_deg,
                gripper_state=gripper_state,
            )
        )

    return resampled_points


def resample_trajectory(
    points: list[TrajectoryPoint],
    *,
    target_points: int | None,
    mode: str,
) -> list[TrajectoryPoint]:
    if target_points is None:
        return points
    if target_points < 1:
        raise ValueError("--target-points must be at least 1.")

    points_by_episode: dict[int, list[TrajectoryPoint]] = {}
    episode_order: list[int] = []
    for point in points:
        if point.episode not in points_by_episode:
            points_by_episode[point.episode] = []
            episode_order.append(point.episode)
        points_by_episode[point.episode].append(point)

    resampled: list[TrajectoryPoint] = []
    for episode in episode_order:
        episode_points = points_by_episode[episode]
        if len(episode_points) <= target_points:
            resampled.extend(episode_points)
            continue

        if mode == "mean":
            resampled.extend(resample_episode_mean(episode_points, target_points))
            continue

        resampled.extend(resample_episode_interpolate(episode_points, target_points))

    return resampled


def print_preview(points: list[TrajectoryPoint], preview_count: int) -> None:
    count = min(max(1, int(preview_count)), len(points))
    print(f"[OK] Loaded {len(points)} trajectory points.")
    print("[OK] Preview target rows:")
    for index, point in enumerate(points[:count], start=1):
        print(
            f"  [{index}] csv_row={point.csv_row_index} episode={point.episode} "
            f"step={point.step} phase={point.phase} "
            f"target_deg=[{format_values(point.target_deg)}]"
        )

    if len(points) > count:
        last_point = points[-1]
        print(
            "  [...] "
            f"last csv_row={last_point.csv_row_index} episode={last_point.episode} "
            f"step={last_point.step} phase={last_point.phase} "
            f"target_deg=[{format_values(last_point.target_deg)}]"
        )


def print_current_robot_state(mc: MyCobotRemote, timeout: float) -> np.ndarray:
    if not mc.update_state(timeout=timeout):
        raise RuntimeError("Gagal ambil state joint dari robot.")

    current_deg = np.asarray(mc.angles_deg, dtype=np.float64)
    print(f"[OK] Current robot deg: [{format_values(current_deg)}]")
    return current_deg


def apply_gripper_state(
    mc: MyCobotRemote,
    point: TrajectoryPoint,
    *,
    use_gripper: bool,
    gripper_speed: int,
    ack_timeout: float,
) -> None:
    if not use_gripper or point.gripper_state is None:
        return

    ok = mc.set_gripper_state(
        point.gripper_state, speed=gripper_speed, timeout=ack_timeout
    )
    if not ok:
        raise RuntimeError(
            f"Gagal kirim gripper state pada csv row {point.csv_row_index}."
        )


def execute_trajectory(points: list[TrajectoryPoint], args: argparse.Namespace) -> None:
    mc = MyCobotRemote(args.robot_ip, port=args.port, timeout=args.timeout)

    try:
        mc.power_on()
        time.sleep(0.5)

        current_deg = print_current_robot_state(mc, timeout=args.timeout)
        first_target_deg = points[0].target_deg
        first_error = wrap_joint_error_deg(current_deg, first_target_deg)
        max_jump = float(np.max(np.abs(first_error)))
        if max_jump > float(args.max_jump_deg):
            raise RuntimeError(
                "Perbedaan pose awal robot ke target CSV pertama terlalu besar: "
                f"{max_jump:.3f} deg > {float(args.max_jump_deg):.3f} deg. "
                "Mulai dari row yang lebih dekat atau naikkan --max-jump-deg jika memang sengaja."
            )

        for index, point in enumerate(points, start=1):
            print(
                f"[{index}/{len(points)}] csv_row={point.csv_row_index} "
                f"episode={point.episode} step={point.step} phase={point.phase}"
            )
            print(f"  target_deg: [{format_values(point.target_deg)}]")
            print(f"  target_rad: [{format_values(point.target_rad)}]")

            ok = mc.send_angles_deg(
                point.target_deg.tolist(),
                speed=args.speed,
                wait=args.wait,
                ack_timeout=args.ack_timeout,
                settle_timeout=args.settle_timeout,
                tolerance_deg=args.tolerance_deg,
            )
            if not ok:
                raise RuntimeError(
                    f"Gagal kirim joint target pada csv row {point.csv_row_index}."
                )

            apply_gripper_state(
                mc,
                point,
                use_gripper=args.use_gripper,
                gripper_speed=args.gripper_speed,
                ack_timeout=args.ack_timeout,
            )

            if args.wait or args.sleep_seconds > 0.0:
                time.sleep(max(0.0, float(args.sleep_seconds)))

            if mc.update_state(timeout=args.timeout):
                current_deg = np.asarray(mc.angles_deg, dtype=np.float64)
                err = wrap_joint_error_deg(current_deg, point.target_deg)
                print(f"  robot_deg : [{format_values(current_deg)}]")
                print(f"  error_deg : [{format_values(err)}]")

        print("[OK] Trajectory execution selesai.")
    finally:
        mc.stop()


def main() -> None:
    args = parse_args()
    csv_path = resolve_csv_path(args.csv)
    original_points = load_trajectory(
        csv_path,
        episode=args.episode,
        phase=args.phase,
        start_step=args.start_step,
        end_step=args.end_step,
        stride=args.stride,
        limit=args.limit,
    )
    points = resample_trajectory(
        original_points,
        target_points=args.target_points,
        mode=args.resample_mode,
    )

    print(f"[OK] CSV: {csv_path}")
    if args.target_points is not None:
        print(
            "[OK] Resampled trajectory "
            f"({args.resample_mode}): {len(original_points)} -> {len(points)} points"
        )
    print_preview(points, preview_count=args.preview_count)

    if not args.execute:
        print(
            "[OK] Preview only. Gunakan --execute untuk mengirim trajectory ke robot."
        )
        return

    execute_trajectory(points, args)


if __name__ == "__main__":
    main()
