import eventlet
eventlet.monkey_patch()

import cv2
import base64
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import numpy as np
import datetime
from paho.mqtt import client as mqtt_client
import random
import time
from collections import deque
from ultralytics import YOLO  # Import YOLOv8

# Flask and SocketIO configuration
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# RTSP camera address
rtsp_url = "rtsp://admin:Mirsmok%2382@192.168.0.13:554/cam/realmonitor?channel=1&subtype=1"

# Flag to control streaming
streaming_enabled = False  # Stream off by default

@app.route('/')
def index():
    return render_template('index2.html')

@socketio.on('connect')
def connect():
    print('Client connected')

# Handle start/stop stream events
@socketio.on('toggle_stream')
def toggle_stream(status):
    global streaming_enabled
    streaming_enabled = status
    print(f"Streaming enabled: {streaming_enabled}")

# MQTT setup
broker = '192.168.0.19'
port = 1883
client_id = f'python-mqtt-{random.randint(0, 1000)}'

def connect_mqtt():
    def on_connect(client, userdata, flags, rc, properties):
        if rc == 0:
            print("Connected to MQTT Broker!")
        else:
            print(f"Failed to connect, return code {rc}\n")

    client = mqtt_client.Client(client_id=client_id, callback_api_version=mqtt_client.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.connect(broker, port)
    return client

def publish(client, topic, msg):
    res = client.publish(topic, msg)
    if res[0] != 0:
        print(f"Failed to send message to topic {topic}")

def process_video():
    frameAnalysisInterval = 40

    # YOLOv8n setup (nano model)
    model = YOLO('yolov8n.pt')  # Load the YOLOv8n (nano) model
    model.classes = [1, 3]  # Wykrywaj tylko wybrane klasy (np. rowery, samochody, autobusy)
    model.conf = 0.3  # Confidence threshold
    model.iou = 0.4  # IOU threshold for NMS

    # Przykład kwantyzacji statycznej modelu
#    model = torch.quantization.quantize_dynamic(
#        model,  # Model do zakwantyzowania
#        {torch.nn.Linear},  # Kwantyzacja tylko warstw liniowych
#        dtype=torch.qint8  # Typ danych kwantyzacji (8-bitowy)
#    )

    video_cap = cv2.VideoCapture(rtsp_url)

    frame_counter = 0
    time_elapsed = datetime.datetime.now()
    car_count_history = deque(maxlen=10)
    stable_car_count = -1
    change_threshold = 0.9

    client = connect_mqtt()
    client.loop_start()

    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(100, 3), dtype="uint8")

    while True:
        eventlet.sleep(0.01)  # Add a small sleep to prevent high CPU usage

        ret, frame = video_cap.read()
        if not ret:
            print("End of the video stream...")
            break

        if frame_counter < frameAnalysisInterval:
            frame_counter += 1
            continue
        frame_counter = 0

        start = datetime.datetime.now()

        # Perform inference with YOLOv8
        results = model(frame)  # Detect objects

        # Dictionary to store counts of each detected class
        class_counts = {}

        # Convert YOLOv8 results to format compatible with OpenCV
        car_count = 0
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes
            confs = result.boxes.conf.cpu().numpy()  # Confidence scores
            clss = result.boxes.cls.cpu().numpy()    # Class indices

            for box, conf, cls in zip(boxes, confs, clss):
                x1, y1, x2, y2 = map(int, box)
                label = f"{model.names[int(cls)]}: {conf:.2f}"
                color = [int(c) for c in colors[int(cls)]] # colors[int(cls)]#[int(c) for c in np.random.randint(0, 255, size=3)]  # Random color for bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1 +10, y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Update class count
                class_name = model.names[int(cls)]
                if class_name in class_counts:
                    class_counts[class_name] += 1
                else:
                    class_counts[class_name] = 1

                # If the object is a car, count it
                if model.names[int(cls)] == 'car':
                    car_count += 1

        car_count_history.append(car_count)
        average_car_count = sum(car_count_history) / len(car_count_history)

        if abs(average_car_count - stable_car_count) > change_threshold:
            stable_car_count = round(average_car_count)


     #   cv2.putText(frame, f"raw cars: {car_count}", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
     #   cv2.putText(frame, f"cars: {stable_car_count}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Display object count below the car count
        y_position = 75
        for class_name, count in class_counts.items():
            cv2.putText(frame, f"{class_name}: {count}", (10, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0),
                        2)
            y_position += 25

        if (datetime.datetime.now() - time_elapsed).total_seconds() > 10.0:
            publish(client, 'device/cctv/carCount', str(stable_car_count))
            time_elapsed = datetime.datetime.now()

        stop = datetime.datetime.now()

        # Display the loop time on the frame

        loop_time = (stop - start).total_seconds()
        text = f"loop: {loop_time *1000:.0f}ms"
        cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if streaming_enabled:
            # Encode image to base64
            _, buffer = cv2.imencode('.jpg', frame)
            frame_encoded = base64.b64encode(buffer).decode('utf-8')
            socketio.emit('frame', {'image': frame_encoded})

        print(f"Loop time: {(stop - start).total_seconds()}")



    video_cap.release()
    client.loop_stop()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    socketio.start_background_task(process_video)
    socketio.run(app, host='0.0.0.0', port=5002)
