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
    conf_threshold = 0.3
    frameAnalysisInterval = 100

    # MobileNet-SSD setup
    net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'mobilenet_iter_73000.caffemodel')

    class_names = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
                   "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
                   "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(class_names), 3), dtype="uint8")

    video_cap = cv2.VideoCapture(rtsp_url)

    frame_counter = 0
    time_elapsed = datetime.datetime.now()
    car_count_history = deque(maxlen=10)
    stable_car_count = -1
    change_threshold = 0.9

    client = connect_mqtt()
    client.loop_start()

    while True:

        ret, frame = video_cap.read()
        if not ret:
            print("End of the video stream...")
            break
        if frame_counter < frameAnalysisInterval:
            frame_counter += 1
            continue
        frame_counter = 0

        mean = np.mean(frame)
        print(f"średnia {mean}")

        start = datetime.datetime.now()

        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (576, 702)), 0.007843, (576, 702), mean)
        net.setInput(blob)
        detections = net.forward()

        # Dictionary to store counts of each detected class
        class_counts = {}

        car_count = 0

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > conf_threshold:
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                label = class_names[idx]

                # Draw the bounding box and label on the frame
                color = [int(c) for c in colors[idx]]
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                cv2.putText(frame, f"{label}: {confidence:.2f}",
                            (startX +5, startY + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Update class count
                class_name = label
                if class_name in class_counts:
                    class_counts[class_name] += 1
                else:
                    class_counts[class_name] = 1

                if label == 'car':
                    car_count += 1

        car_count_history.append(car_count)
        average_car_count = sum(car_count_history) / len(car_count_history)

        if abs(average_car_count - stable_car_count) > change_threshold:
            stable_car_count = round(average_car_count)

        # Display object count below the car count
        y_position = 75
        for class_name, count in class_counts.items():
            cv2.putText(frame, f"{class_name}: {count}", (10, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            y_position += 25

        if (datetime.datetime.now() - time_elapsed).total_seconds() > 10.0:
            publish(client, 'device/cctv/carCount', str(stable_car_count))
            time_elapsed = datetime.datetime.now()

        stop = datetime.datetime.now()
        loop_time = (stop-start).total_seconds()
        text = f"loop: {loop_time *1000:.0f}ms"
        cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if streaming_enabled:
            # Encode image to base64
            _, buffer = cv2.imencode('.jpg', frame)
            frame_encoded = base64.b64encode(buffer).decode('utf-8')
            socketio.emit('frame', {'image': frame_encoded})

        stop = datetime.datetime.now()
        print(f"Loop time: {(stop-start).total_seconds()}")

        eventlet.sleep(0.5)  # Add a small sleep to prevent high CPU usage

    video_cap.release()
    client.loop_stop()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    socketio.start_background_task(process_video)
    socketio.run(app, host='0.0.0.0', port=5002)
