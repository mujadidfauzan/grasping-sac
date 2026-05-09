import socket
import time

RASPI_IP = "10.16.120.250"
UDP_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


def kirim_ke_robot(j1, j2, j3, j4, j5, j6):
    pesan = f"{j1},{j2},{j3},{j4},{j5},{j6}"
    sock.sendto(pesan.encode(), (RASPI_IP, UDP_PORT))


try:
    print("Mengirim perintah gerak ke myCobot...")

    print("Gerakan 1...")
    kirim_ke_robot(45, 0, 0, 0, 0, 0)
    time.sleep(2)

    print("Gerakan 2...")
    kirim_ke_robot(0, -20, -20, 0, 0, 0)
    time.sleep(2)

    print("Kembali ke home...")
    kirim_ke_robot(0, 0, 0, 0, 0, 0)

except Exception as e:
    print(f"Error: {e}")
finally:
    print("Selesai.")
