from __future__ import annotations

import argparse
import time

import numpy as np

from sim2real.remote import MyCobotRemote

DEFAULT_ROBOT_IP = "10.42.0.1"
DEFAULT_PORT = 5005
DEFAULT_SPEED = 20
DEFAULT_TIMEOUT = 2.0
DEFAULT_ACK_TIMEOUT = 5.0
DEFAULT_SETTLE_TIMEOUT = 20.0
DEFAULT_TOLERANCE_DEG = 3.0

SAMPLE_MOVES_RAD = [
    # ("home", [0.0, 0.0, 0.0, 0, 0.0, 0.0]),
    ("sample_1", [0.0, -1.0, 0.0, 0.0, 0.0, 0.0]),
    # ("sample_2", [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
    # ("sample_3", [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
    # ("sample_4", [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
    # ("sample_5", [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
    # ("sample_6", [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
    ("home", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simple laptop test for MyCobot bridge. Input joint targets in radians."
    )
    parser.add_argument("--robot-ip", default=DEFAULT_ROBOT_IP)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--speed", type=int, default=DEFAULT_SPEED)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--ack-timeout", type=float, default=DEFAULT_ACK_TIMEOUT)
    parser.add_argument("--settle-timeout", type=float, default=DEFAULT_SETTLE_TIMEOUT)
    parser.add_argument("--tolerance-deg", type=float, default=DEFAULT_TOLERANCE_DEG)
    parser.add_argument("--skip-motion", action="store_true")
    return parser.parse_args()


def format_values(values: list[float]) -> str:
    return ", ".join(f"{value:.3f}" for value in values)


def main() -> None:
    args = parse_args()
    mc = MyCobotRemote(args.robot_ip, port=args.port, timeout=args.timeout)

    try:
        mc.power_on()
        time.sleep(0.5)

        if not mc.update_state(timeout=args.timeout):
            raise RuntimeError("Gagal ambil state dari robot.")

        print(f"Angles awal degree: [{format_values(mc.angles_deg)}]")
        print(f"Angles awal radian: [{format_values(mc.angles_rad)}]")
        print(f"Coords awal: [{format_values(mc.coords)}]")

        if args.skip_motion:
            print("Skip motion aktif. Test state selesai.")
            return

        for name, target_rad in SAMPLE_MOVES_RAD:
            target_deg = np.rad2deg(np.asarray(target_rad, dtype=np.float64)).tolist()
            print(f"\nTarget `{name}` radian: [{format_values(target_rad)}]")
            print(f"Target `{name}` degree: [{format_values(target_deg)}]")

            ok = mc.send_angles_rad(
                target_rad,
                speed=args.speed,
                wait=False,
                ack_timeout=args.ack_timeout,
                settle_timeout=args.settle_timeout,
                tolerance_deg=args.tolerance_deg,
            )
            if not ok:
                raise RuntimeError(f"Gagal kirim pose `{name}` ke robot.")

            mc.update_state(timeout=args.timeout)
            print(f"Angles robot degree: [{format_values(mc.angles_deg)}]")
            print(f"Angles robot radian: [{format_values(mc.angles_rad)}]")
            print(f"Coords robot: [{format_values(mc.coords)}]")
            time.sleep(1.0)

        print("\nTest koneksi dan kirim sample angles selesai.")
    finally:
        mc.stop()


if __name__ == "__main__":
    main()
