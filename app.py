from flask import Flask, render_template, jsonify
import cv2
import numpy as np
import base64
import threading

app = Flask(__name__)
cap = cv2.VideoCapture(0)
movement_detected = False

def detect_motion():
    
    ret, frame1 = cap.read()

    ret, frame2 = cap.read()

    while cap.isOpened():
        diff = cv2.absdiff(frame1, frame2)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 30, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=3)
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        movement_detected = False

        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            if cv2.contourArea(contour) < 2000:
                continue
            cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
           
            movement_detected = True
            break

        if movement_detected:
            status = "Movement Detected"
        else:
            status = "No Movement"

        # Encode the frame as JPEG
        _, frame_encoded = cv2.imencode('.jpg', frame1)
        frame_bytes = base64.b64encode(frame_encoded).decode('utf-8')

        # Send the status and frame to the client
        send_status(status)
        send_frame(frame_bytes)

        frame1 = frame2
        ret, frame2 = cap.read()

        if cv2.waitKey(40) == 27:
            break

    cv2.destroyAllWindows()
    cap.release()

def send_status(status):
    global current_status
    current_status = status

@app.route('/')
def index():
    return render_template('welcome.html')

@app.route('/startgame')
def startgame():
    return render_template('index.html')

@app.route('/status')
def get_status():
    return jsonify({"status": current_status})

def send_frame(frame):
    global current_frame
    current_frame = frame

@app.route('/frame')
def get_frame():
    return jsonify({"frame": current_frame})

@app.route('/start_detection')
def start_detection():
    threading.Thread(target=detect_motion).start()
    return jsonify({"status": "Detection started"})

@app.route('/restart_camera_and_tracking')
def restart_camera_and_tracking():
    global current_status
    current_status = ""  # Reset status
    global current_frame
    current_frame = ""   # Reset frame
    threading.Thread(target=detect_motion).start()  # Restart detection
    return jsonify({"status": "Camera and tracking restarted"})

@app.route('/gameover')
def gameover():
    
    return render_template('gameover.html')


    



@app.route('/winner')
def winner():
    return render_template('winner.html')

if __name__ == '__main__':
    current_status = ""  # Variable to store current status
    current_frame = ""   # Variable to store current frame
    app.run(debug=True)
