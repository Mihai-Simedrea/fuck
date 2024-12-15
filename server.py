import socket
from PIL import Image
import io

HOST = "0.0.0.0"
PORT = 12345

def main():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind((HOST, PORT))
        server_socket.listen(1)
        print(f"Server listening on {HOST}:{PORT}")

        while True:
            client_socket, addr = server_socket.accept()
            print(f"Connection from {addr}")

            with client_socket:
                data = bytearray()
                while True:
                    packet = client_socket.recv(4096)
                    if not packet:
                        break
                    data.extend(packet)

                try:
                    image = Image.open(io.BytesIO(data))
                    image.show()
                except Exception as e:
                    print(f"Error decoding image: {e}")

if __name__ == "__main__":
    main()
