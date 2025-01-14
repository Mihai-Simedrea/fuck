import socket
import threading
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter
from brainflow.ml_model import MLModel, BrainFlowMetrics, BrainFlowClassifiers, BrainFlowModelParams
import time

def send_focus_data(client_socket):
    params = BrainFlowInputParams()
    params.serial_port = '/dev/ttyUSB0'
    params.serial_number = 'UN-2019.06.23'
    board = BoardShim(8, params)
    board.prepare_session()

    try:
        board.start_stream()
        master_board_id = board.get_board_id()
        sampling_rate = BoardShim.get_sampling_rate(master_board_id)

        mindfulness_params = BrainFlowModelParams(BrainFlowMetrics.MINDFULNESS.value,
                                              BrainFlowClassifiers.DEFAULT_CLASSIFIER.value)
        mindfulness = MLModel(mindfulness_params)
        mindfulness.prepare()

        while True:
            time.sleep(2)
            data = board.get_board_data()
            eeg_channels = BoardShim.get_eeg_channels(int(master_board_id))
            bands = DataFilter.get_avg_band_powers(data, eeg_channels, sampling_rate, True)
            feature_vector = bands[0]

            prediction = mindfulness.predict(feature_vector)
            focus = 0 if prediction > 0.5 else 1

            print(f'Focus: {focus}')
            
            client_socket.send(str(focus).encode())

    except Exception as e:
        print(f"Error: {e}")
    finally:
        board.stop_stream()
        board.release_session()

def handle_client(client_socket, client_address):
    print(f"New connection: {client_address}")
    try:
        send_focus_data(client_socket)
    except Exception as e:
        print(f"Client {client_address} error: {e}")
    finally:
        client_socket.close()

def start_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(('0.0.0.0', 1235))
    server.listen(5)

    print("Server is running... Waiting for connections.")

    while True:
        client_socket, client_address = server.accept()
        client_handler = threading.Thread(target=handle_client, args=(client_socket, client_address))
        client_handler.start()

if __name__ == "__main__":
    start_server()
