from flask import Flask, render_template, request, Response
from ultralytics import YOLO
import cv2
import os

app = Flask(__name__)

# Load helmet model
model = YOLO("model/helmet.pt")

# Global variable to track helmet detection status
helmet_status = True  # True = helmet detected (GREEN), False = no helmet (RED)

@app.route("/", methods=["GET", "POST"])
def index():
    signal = None

    if request.method == "POST":
        image = request.files["image"]
        img_path = "static/upload.jpg"
        image.save(img_path)

        results = model(img_path)

        helmet_detected = False

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])

                # Class 0 = helmet (common in these models)
                if cls == 0:
                    helmet_detected = True

        if helmet_detected:
            signal = "GREEN"
        else:
            signal = "RED"

    return render_template("index.html", signal=signal)

def generate_frames():
    global helmet_status
    camera = cv2.VideoCapture(0)
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # Run helmet detection
        results = model(frame, conf=0.5)
        
        all_wearing_helmets = True  # Assume all are wearing helmets
        no_helmet_detected = False  # Track if anyone is NOT wearing helmet
        person_count = 0
        
        # Draw boxes and check for helmets
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                person_count += 1
                
                if cls == 0:  # Helmet detected
                    color = (0, 255, 0)  # Green box
                    label = "Helmet"
                else:  # No helmet detected
                    color = (0, 0, 255)  # Red box
                    label = "No Helmet"
                    no_helmet_detected = True
                    all_wearing_helmets = False
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Update global status - GREEN only if ALL are wearing helmets
        # RED if ANYONE is not wearing helmet
        helmet_status = all_wearing_helmets and person_count > 0
        
        # Add signal indicator pixel in top-left corner
        # GREEN = all wearing helmets, RED = anyone not wearing helmet or no detections
        if no_helmet_detected or person_count == 0:
            cv2.rectangle(frame, (20, 40), (21, 41), (255, 0, 0), -1)  # Red pixel - PAUSE TIMER
        else:
            cv2.rectangle(frame, (20, 40), (21, 41), (0, 255, 0), -1)  # Green pixel - CONTINUE TIMER
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
