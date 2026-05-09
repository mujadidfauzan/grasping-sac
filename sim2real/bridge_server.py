import json
import socket

from pymycobot import MyCobot280

# Inisialisasi Robot
mc = MyCobot280("/dev/ttyAMA0", 115200)

# Setup Socket UDP
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", 5005))

print("Bridge MyCobot Aktif [Port 5005]...")


def send_json(addr, payload):
    sock.sendto(json.dumps(payload).encode(), addr)


def parse_angles_payload(data):
    if isinstance(data, dict):
        return data.get("angles"), int(data.get("speed", 50)), data.get("seq")
    return data, 50, None


def parse_gripper_payload(data):
    if isinstance(data, dict):
        return int(data.get("state", 0)), int(data.get("speed", 50))
    return int(data), 50

while True:
    data, addr = sock.recvfrom(1024)
    cmd = None

    try:
        payload = json.loads(data.decode())

        # CEK REQUEST DARI LAPTOP
        cmd = payload.get("command")

        if cmd == "GET_STATE":
            # Ambil data dari robot
            res = {
                "response": "STATE",
                "ok": True,
                "angles": mc.get_angles(),
                "coords": mc.get_coords(),
            }
            send_json(addr, res)

        elif cmd == "SET_ANGLES":
            # Terima sudut dan gerakkan
            angles, speed, seq = parse_angles_payload(payload.get("data"))
            if not isinstance(angles, list):
                raise ValueError("SET_ANGLES membutuhkan list joint angles.")

            print(f"SET_ANGLES seq={seq}: {angles} @ speed={speed}")
            mc.send_angles(angles, speed)
            send_json(
                addr,
                {"response": "SET_ANGLES_ACK", "ok": True, "seq": seq, "speed": speed},
            )

        elif cmd == "SET_GRIPPER":
            # Kontrol Gripper (1=tutup, 0=buka)
            state, speed = parse_gripper_payload(payload.get("data"))
            mc.set_gripper_state(state, speed)
            mc.set_gripper_state(state, speed)
            send_json(
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
        send_json(addr, {"response": response_type, "ok": False, "error": str(exc)})
