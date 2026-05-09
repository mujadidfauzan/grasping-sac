from __future__ import annotations

import json
import socket
import time

import numpy as np


class MyCobotRemote:
    DEFAULT_PORT = 5005
    DEFAULT_TIMEOUT = 2.0
    DEFAULT_ACK_TIMEOUT = 5.0
    DEFAULT_SETTLE_TIMEOUT = 20.0
    DEFAULT_POLL_INTERVAL = 0.2
    NUM_JOINTS = 6
    RESPONSE_TYPES = {
        "GET_STATE": "STATE",
        "SET_ANGLES": "SET_ANGLES_ACK",
        "SET_GRIPPER": "SET_GRIPPER_ACK",
    }

    def __init__(
        self, ip: str, port: int = DEFAULT_PORT, timeout: float = DEFAULT_TIMEOUT
    ):
        self.addr = (ip, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(timeout)
        self._next_seq = 0
        self._last_angles_deg: list[float] = [0.0] * self.NUM_JOINTS
        self._last_coords: list[float] = [0.0] * self.NUM_JOINTS

    def _send(
        self,
        command: str,
        data=None,
        *,
        timeout: float | None = None,
    ) -> dict | None:
        message = json.dumps({"command": command, "data": data}).encode()
        original_timeout = self.sock.gettimeout()
        expected_response = self.RESPONSE_TYPES.get(command)

        try:
            self.sock.sendto(message, self.addr)
            self.sock.settimeout(original_timeout if timeout is None else float(timeout))

            while True:
                raw, _ = self.sock.recvfrom(4096)
                response = json.loads(raw.decode())
                if (
                    expected_response is None
                    or response.get("response") == expected_response
                ):
                    return response
        except (OSError, socket.timeout, json.JSONDecodeError):
            return None
        finally:
            self.sock.settimeout(original_timeout)

    def update_state(self, timeout: float | None = None) -> bool:
        response = self._send("GET_STATE", timeout=timeout)
        if not response or not response.get("ok"):
            return False

        angles = response.get("angles")
        coords = response.get("coords")
        if isinstance(angles, list) and len(angles) == self.NUM_JOINTS:
            self._last_angles_deg = [float(value) for value in angles]
        if isinstance(coords, list) and len(coords) == self.NUM_JOINTS:
            self._last_coords = [float(value) for value in coords]

        return True

    @property
    def angles_deg(self) -> list[float]:
        return self._last_angles_deg

    @property
    def angles_rad(self) -> list[float]:
        return np.deg2rad(np.asarray(self._last_angles_deg, dtype=np.float64)).tolist()

    @property
    def angles(self) -> list[float]:
        return self.angles_deg

    @property
    def coords(self) -> list[float]:
        return self._last_coords

    def get_angles(self) -> list[float]:
        self.update_state()
        return self.angles_deg

    def get_coords(self) -> list[float]:
        self.update_state()
        return self.coords

    @staticmethod
    def _max_joint_error_deg(current: list[float], target: list[float]) -> float:
        errors = []
        for current_angle, target_angle in zip(current, target):
            delta = (float(current_angle) - float(target_angle) + 180.0) % 360.0 - 180.0
            errors.append(abs(delta))
        return max(errors, default=float("inf"))

    def wait_until_angles_reached(
        self,
        angles_deg: list[float],
        *,
        tolerance_deg: float = 2.0,
        timeout: float = DEFAULT_SETTLE_TIMEOUT,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
    ) -> bool:
        deadline = time.monotonic() + float(timeout)
        target = [float(value) for value in angles_deg]

        while time.monotonic() <= deadline:
            if self.update_state(timeout=poll_interval):
                if self._max_joint_error_deg(self._last_angles_deg, target) <= tolerance_deg:
                    return True
            time.sleep(poll_interval)

        return False

    def send_angles_deg(
        self,
        angles_deg: list[float],
        speed: int = 20,
        *,
        wait: bool = False,
        ack_timeout: float = DEFAULT_ACK_TIMEOUT,
        settle_timeout: float = DEFAULT_SETTLE_TIMEOUT,
        tolerance_deg: float = 2.0,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
    ) -> bool:
        target = [float(value) for value in angles_deg]
        seq = self._next_seq
        self._next_seq += 1

        response = self._send(
            "SET_ANGLES",
            {"angles": target, "speed": int(speed), "seq": seq},
            timeout=ack_timeout,
        )
        if not response or not response.get("ok"):
            return False

        if wait:
            return self.wait_until_angles_reached(
                target,
                tolerance_deg=tolerance_deg,
                timeout=settle_timeout,
                poll_interval=poll_interval,
            )
        return True

    def send_angles_rad(
        self,
        angles_rad: list[float],
        speed: int = 20,
        *,
        wait: bool = False,
        ack_timeout: float = DEFAULT_ACK_TIMEOUT,
        settle_timeout: float = DEFAULT_SETTLE_TIMEOUT,
        tolerance_deg: float = 2.0,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
    ) -> bool:
        angles_deg = np.rad2deg(np.asarray(angles_rad, dtype=np.float64)).tolist()
        return self.send_angles_deg(
            angles_deg,
            speed=speed,
            wait=wait,
            ack_timeout=ack_timeout,
            settle_timeout=settle_timeout,
            tolerance_deg=tolerance_deg,
            poll_interval=poll_interval,
        )

    def send_angles(
        self,
        angles: list[float],
        speed: int = 20,
        *,
        wait: bool = False,
        ack_timeout: float = DEFAULT_ACK_TIMEOUT,
        settle_timeout: float = DEFAULT_SETTLE_TIMEOUT,
        tolerance_deg: float = 2.0,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
    ) -> bool:
        return self.send_angles_deg(
            angles,
            speed=speed,
            wait=wait,
            ack_timeout=ack_timeout,
            settle_timeout=settle_timeout,
            tolerance_deg=tolerance_deg,
            poll_interval=poll_interval,
        )

    def set_gripper_state(
        self, state: int, speed: int = 50, timeout: float = DEFAULT_ACK_TIMEOUT
    ) -> bool:
        response = self._send(
            "SET_GRIPPER",
            {"state": int(state), "speed": int(speed)},
            timeout=timeout,
        )
        return bool(response and response.get("ok"))

    def power_on(self) -> None:
        print(f"Connected to robot at {self.addr}")

    def stop(self) -> None:
        print("Stopping remote connection...")
        self.sock.close()
