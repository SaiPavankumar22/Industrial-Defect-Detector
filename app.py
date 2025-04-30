from flask import Flask, request, jsonify, send_file, render_template, session
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
import base64
from ultralytics import YOLO
import requests
import supervision as sv
from datetime import datetime, timedelta
import tempfile
import uuid
import time
from flask_session import Session

app = Flask(__name__)
CORS(app)

# Session configuration
app.config['SECRET_KEY'] = 'sai_22'
app.config['SESSION_TYPE'] = 'filesystem'
app.config['PERMANENT_SESSION_LIFETIME'] = 3600  # 1 hour
app.config['SESSION_FILE_DIR'] = os.path.join(os.getcwd(), 'flask_session')

# Ensure session directory exists
os.makedirs(app.config['SESSION_FILE_DIR'], exist_ok=True)

# Initialize session
Session(app)

# Add timestamp function to template context
@app.context_processor
def utility_processor():
    def get_timestamp():
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return dict(now=get_timestamp)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi', 'mov'}
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# Dictionary to store temporary results
temporary_results = {}

def generate_result_id():
    return str(uuid.uuid4())

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('open.html')

@app.route('/image')
def image():
    return render_template('image.html')

@app.route('/video')
def video():
    return render_template('video.html')

@app.route('/prediction')
def predictions():
    return render_template('prediction.html')

@app.route('/video_process', methods=['POST'])
def process_video():
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file uploaded'}), 400

        video_file = request.files['video']
        if not allowed_file(video_file.filename):
            return jsonify({'error': 'Invalid file type'}), 400

        # Save video temporarily
        temp_dir = tempfile.mkdtemp()
        video_path = os.path.join(temp_dir, secure_filename(video_file.filename))
        video_file.save(video_path)

        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return jsonify({'error': 'Failed to open video file'}), 500

        frames = []
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Extract one frame every second
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process every nth frame (n = fps to get 1 frame per second)
            if frame_count % fps == 0:
                # Convert frame to base64
                _, buffer = cv2.imencode('.jpg', frame)
                base64_frame = base64.b64encode(buffer).decode('utf-8')
                
                # Calculate timestamp
                seconds = frame_count // fps
                timestamp = str(timedelta(seconds=seconds))
                
                frames.append({
                    'id': len(frames) + 1,
                    'src': f"data:image/jpeg;base64,{base64_frame}",
                    'timestamp': timestamp
                })

            frame_count += 1
            if frame_count >= total_frames:
                break

        cap.release()
        
        # Clean up
        os.remove(video_path)
        os.rmdir(temp_dir)

        return jsonify({
            'frames': frames,
            'total_frames': total_frames,
            'fps': fps
        })

    except Exception as e:
        print("Error processing video:", str(e))
        return jsonify({'error': 'Error processing video'}), 500
    


@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("🔥 Starting prediction process...")
        # Check if an image file and model type are provided
        if 'image' not in request.files or 'model_type' not in request.form:
            print("🔥 Error: Missing image or model type")
            return jsonify({'error': 'Image file or model type missing'}), 400

        file = request.files['image']
        model_type = request.form['model_type']
        print(f"🔥 Received file: {file.filename}, model type: {model_type}")

        if not file or file.filename == '':
            print("🔥 Error: No file selected")
            return jsonify({'error': 'No selected file'}), 400

        if not allowed_file(file.filename):
            print("🔥 Error: Invalid file type")
            return jsonify({'error': 'Invalid file type'}), 400

        # Save uploaded image
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        print(f"🔥 Saved uploaded file to: {filepath}")

        try:
            # Load appropriate model and class labels based on model_type
            print(f"🔥 Loading model for type: {model_type}")
            if model_type == "gap-measurement":
                model = YOLO("gap_final.pt")
            elif model_type == "weld-defect":
                model = YOLO("weld_yolov8m-obb.pt")
            elif model_type == "paint-aesthetics":
                model = YOLO("paint_best.pt")
            else:
                print("🔥 Error: Invalid model type")
                return jsonify({'error': 'Invalid model type selected'}), 400

            # Run YOLO inference with confidence threshold
            print("🔥 Running YOLO inference...")
            results = model(filepath, conf=0.25)  # Adjusted confidence threshold
            detections = sv.Detections.from_ultralytics(results[0])
            print(f"🔥 Detections: {len(detections)} objects found")

            # Load the image
            image = cv2.imread(filepath)
            if image is None:
                print("🔥 Error: Failed to load image")
                return jsonify({'error': 'Failed to load image'}), 500

            # Annotate image
            box_annotator = sv.BoxAnnotator()
            
            # Use model's native class names
            labels = [
                f"{class_name} {confidence:.2f}"
                for class_name, confidence
                in zip(detections.data['class_name'], detections.confidence)
            ]
            
            print(f"🔥 Generated labels: {labels}")
            annotated_frame = box_annotator.annotate(
                scene=image,
                detections=detections
            )

            # Add text labels manually with improved visibility
            for i, label in enumerate(labels):
                xyxy = detections.xyxy[i]
                x1, y1 = int(xyxy[0]), int(xyxy[1])

                # Add background to text for better visibility
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(
                    annotated_frame,
                    (x1, y1 - text_size[1] - 10),
                    (x1 + text_size[0], y1),
                    (0, 0, 0),
                    -1
                )

                cv2.putText(
                    annotated_frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

            # Save the result
            result_filename = 'result_' + filename
            result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
            cv2.imwrite(result_path, annotated_frame)
            print(f"🔥 Saved annotated image to: {result_path}")

            # Collect unique defect class names
            unique_labels = list(set(detections.data['class_name']))
            print(f"🔥 Unique labels found: {unique_labels}")

            # Generate LLM explanation
            if unique_labels:
                defect_summary = ', '.join(unique_labels)
                print(f"🔥 Generating explanation for: {defect_summary}")
                reason = ask_groq(f"Explain why {defect_summary} occur as defects in any components for {model_type}. give me the explanation in the form of why it is happening and how to avoid it.")
                print(f"🔥 Generated explanation: {reason}")
            else:
                defect_summary = "No defect detected"
                reason = "No defect was identified by the model."
                print("🔥 No defects detected")

            # Read the result image and convert to base64
            with open(result_path, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
                image_url = f"data:image/jpeg;base64,{encoded_image}"

            # Read and encode the original image
            with open(filepath, "rb") as original_file:
                encoded_original = base64.b64encode(original_file.read()).decode('utf-8')
                original_image_url = f"data:image/jpeg;base64,{encoded_original}"

            # Return the results directly
            return jsonify({
                'defect': defect_summary,
                'reason': reason,
                'image': image_url,
                'original_image': original_image_url,
                'session_id': session.get('result_id', 'Not available')
            })

        except Exception as e:
            print(f"🔥 Model Processing Error: {str(e)}")
            return jsonify({'error': f'Error processing model: {str(e)}'}), 500

    except Exception as e:
        print(f"🔥 Backend Error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/prediction')
def prediction():
    try:
        print("🔥 Loading prediction page...")
        print(f"🔥 Session data: {dict(session)}")
        
        result_id = session.get('result_id')
        print(f"🔥 Retrieved result_id from session: {result_id}")
        
        if not result_id:
            print("🔥 No result_id found in session")
            return render_template('prediction.html', 
                                error='No prediction results available. Please upload an image first.',
                                defect='No defect information available',
                                reason='No explanation available',
                                image='')

        if result_id not in temporary_results:
            print(f"🔥 Result ID {result_id} not found in temporary_results")
            print(f"🔥 Available results: {list(temporary_results.keys())}")
            return render_template('prediction.html', 
                                error='Prediction results expired. Please upload a new image.',
                                defect='No defect information available',
                                reason='No explanation available',
                                image='')

        result = temporary_results[result_id]
        print(f"🔥 Retrieved result from temporary_results: {result}")
        
        # Read the result image
        result_path = os.path.join(app.config['UPLOAD_FOLDER'], result['result_filename'])
        print(f"🔥 Reading result image from: {result_path}")
        
        if not os.path.exists(result_path):
            print(f"🔥 Error: Result image file not found at {result_path}")
            return render_template('prediction.html', 
                                error='Result image not found.',
                                defect=result['defect'],
                                reason=result['reason'],
                                image='')
        
        try:
            with open(result_path, "rb") as image_file:
                image_data = image_file.read()
                print(f"🔥 Read {len(image_data)} bytes from image file")
                encoded_image = base64.b64encode(image_data).decode('utf-8')
                print("🔥 Successfully encoded image to base64")
            
            # Create the data URL
            image_url = f"data:image/jpeg;base64,{encoded_image}"
            print(f"🔥 Created image URL with length: {len(image_url)}")
            
            return render_template('prediction.html', 
                                defect=result['defect'],
                                reason=result['reason'],
                                image=image_url)
        except Exception as e:
            print(f"🔥 Error processing image: {str(e)}")
            import traceback
            print(f"🔥 Stack trace: {traceback.format_exc()}")
            return render_template('prediction.html', 
                                error='Error processing result image.',
                                defect=result['defect'],
                                reason=result['reason'],
                                image='')
            
    except Exception as e:
        print(f"🔥 Prediction Page Error: {str(e)}")
        import traceback
        print(f"🔥 Stack trace: {traceback.format_exc()}")
        return render_template('prediction.html', 
                            error='An error occurred while loading prediction results.',
                            defect='Error',
                            reason='Please try again.',
                            image='')

# Clean up old results periodically
def cleanup_old_results():
    # Remove results older than 1 hour
    current_time = time.time()
    for result_id, result in list(temporary_results.items()):
        if current_time - result.get('timestamp', 0) > 3600:  # 1 hour
            try:
                result_path = os.path.join(app.config['UPLOAD_FOLDER'], result['result_filename'])
                if os.path.exists(result_path):
                    os.remove(result_path)
                del temporary_results[result_id]
            except Exception as e:
                print(f"Error cleaning up result {result_id}: {str(e)}")

# Schedule cleanup
import threading
import time

def cleanup_loop():
    while True:
        cleanup_old_results()
        time.sleep(3600)  # Run every hour

# Start cleanup thread
cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
cleanup_thread.start()

def ask_groq(prompt):
    payload = {
        "model": "llama3-70b-8192",
        "messages": [
            {"role": "system", "content": "You are an expert on mechanical engineering who deals with many fields such as defects in welding, gap measurement, paint asthetics."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 300
    }
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    try:
        response = requests.post(GROQ_URL, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except requests.exceptions.RequestException as e:
        print("Groq API Error:", e)
        return "Reason unavailable due to API error."



if __name__ == '__main__':
    app.run(debug=True, port=5000, use_reloader=False)