import socket
import cv2
import numpy as np
from fer import FER

HOST = "0.0.0.0"
PORT = 12345
detector = FER(mtcnn=True)

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
                    nparr = np.frombuffer(data, np.uint8)
                    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if image is not None:
                        cv2.waitKey(1)
                        result = detector.detect_emotions(image)
                        print(result)
                    else:
                        print("Error decoding image")
                except Exception as e:
                    print(f"Error: {e}")

if __name__ == "__main__":
    main()
