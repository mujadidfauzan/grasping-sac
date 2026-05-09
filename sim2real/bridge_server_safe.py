from __future__ import annotations

import argparse
import json
import socket

from pymycobot import MyCobot280

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 5005
DEFAULT_SERIAL_PORT = "/dev/ttyAMA0"
DEFAULT_BAUD = 115200
DEFAULT_MAX_SPEED = 100
DEFAULT_BUFFER_SIZE = 4096


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simple UDP bridge from laptop to MyCobot."
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--serial-port", default=DEFAULT_SERIAL_PORT)
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    parser.add_argument("--max-speed", type=int, default=DEFAULT_MAX_SPEED)
    return parser.parse_args()


def send_json(sock: socket.socket, addr, payload: dict) -> None:
    sock.sendto(json.dumps(payload).encode(), addr)


def sanitize_speed(speed: int | float, max_speed: int) -> int:
    return max(1, min(int(speed), int(max_speed)))


def get_robot_state(mc: MyCobot280) -> tuple[list[float], list[float]]:
    angles = mc.get_angles()
    coords = mc.get_coords()
    if not isinstance(angles, list) or len(angles) != 6:
        raise ValueError(f"Invalid robot joint state: {angles}")
    if not isinstance(coords, list) or len(coords) != 6:
        raise ValueError(f"Invalid robot Cartesian state: {coords}")
    return [float(value) for value in angles], [float(value) for value in coords]


def parse_angles_payload(payload: dict) -> tuple[list[float], int, int | None]:
    data = payload.get("data")
    if not isinstance(data, dict):
        raise ValueError("SET_ANGLES data harus berupa object.")

    angles = data.get("angles")
    speed = data.get("speed", 20)
    seq = data.get("seq")

    if not isinstance(angles, list) or len(angles) != 6:
        raise ValueError("SET_ANGLES membutuhkan 6 joint angles.")

    return [float(angle) for angle in angles], int(speed), seq


def parse_gripper_payload(payload: dict) -> tuple[int, int]:
    data = payload.get("data")
    if isinstance(data, dict):
        state = int(data.get("state", 0))
        speed = int(data.get("speed", 50))
    else:
        state = int(data)
        speed = 50

    if state not in (0, 1):
        raise ValueError("SET_GRIPPER state harus 0 atau 1.")
    return state, speed


def main() -> None:
    args = parse_args()
    mc = MyCobot280(args.serial_port, args.baud)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.host, args.port))

    print(
        "Simple bridge active "
        f"[udp={args.host}:{args.port} serial={args.serial_port} baud={args.baud}]"
    )

    try:
        while True:
            raw_data, addr = sock.recvfrom(DEFAULT_BUFFER_SIZE)
            cmd = None

            try:
                payload = json.loads(raw_data.decode())
                cmd = payload.get("command")

                if cmd == "GET_STATE":
                    angles, coords = get_robot_state(mc)
                    send_json(
                        sock,
                        addr,
                        {
                            "response": "STATE",
                            "ok": True,
                            "angles": angles,
                            "coords": coords,
                        },
                    )
                elif cmd == "SET_ANGLES":
                    angles, speed, seq = parse_angles_payload(payload)
                    speed = sanitize_speed(speed, args.max_speed)
                    print(f"SET_ANGLES seq={seq} angles={angles} speed={speed}")
                    mc.send_angles(angles, speed)
                    send_json(
                        sock,
                        addr,
                        {
                            "response": "SET_ANGLES_ACK",
                            "ok": True,
                            "seq": seq,
                            "angles": angles,
                            "speed": speed,
                        },
                    )
                elif cmd == "SET_GRIPPER":
                    state, speed = parse_gripper_payload(payload)
                    speed = sanitize_speed(speed, args.max_speed)
                    mc.set_gripper_state(state, speed)
                    mc.set_gripper_state(state, speed)
                    send_json(
                        sock,
                        addr,
                        {
                            "response": "SET_GRIPPER_ACK",
                            "ok": True,
                            "state": state,
                            "speed": speed,
                        },
                    )
                else:
                    raise ValueError(f"Unknown command: {cmd}")
            except Exception as exc:
                print(f"Bridge error ({cmd}): {exc}")
                response_type = {
                    "GET_STATE": "STATE",
                    "SET_ANGLES": "SET_ANGLES_ACK",
                    "SET_GRIPPER": "SET_GRIPPER_ACK",
                }.get(cmd, "ERROR")
                send_json(
                    sock,
                    addr,
                    {"response": response_type, "ok": False, "error": str(exc)},
                )
    finally:
        sock.close()


if __name__ == "__main__":
    main()
