# app.py
from flask import Flask, render_template, request, Response, jsonify, send_file
import cv2
import os
from werkzeug.utils import secure_filename
import threading
import time
from face_blur import blur_face, initialize_model, process_video
from shared import processing_progress

app = Flask(__name__)

# Configure upload folder (temporary storage)
UPLOAD_FOLDER = 'temp'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure temp folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize the YOLO model
model = initialize_model()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(input_path)
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'blurred_' + filename)
        
        # Start processing in a separate thread
        thread = threading.Thread(target=process_video, args=(input_path, output_path, model, filename))
        thread.start()
        
        return jsonify({'message': 'Processing started', 'filename': filename}), 202
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/progress/<filename>')
def get_progress(filename):
    progress = processing_progress.get(filename, 0)
    return jsonify({'progress': progress})

@app.route('/download/<filename>')
def download_file(filename):
    path = os.path.join(app.config['UPLOAD_FOLDER'], 'blurred_' + filename)
    return send_file(path, as_attachment=True)

@app.route('/camera')
def camera():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames():
    cap = cv2.VideoCapture(0)
    # Set a larger resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('temp/camera_output.mp4', fourcc, 20.0, (1280, 720))
    
    try:
        while True:
            success, frame = cap.read()
            if not success:
                break
            else:
                # Process frame with YOLO and blur faces
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = model(frame_rgb)
                for result in results:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    for box in boxes:
                        frame = blur_face(frame, box)
                
                # Write the frame to the video file
                out.write(frame)
                
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    finally:
        cap.release()
        out.release()

@app.route('/save_camera_video')
def save_camera_video():
    return send_file('temp/camera_output.mp4', as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)