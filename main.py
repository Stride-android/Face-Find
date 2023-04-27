import base64
import os
import threading

from flask import Flask, request, render_template, jsonify
from flask_ngrok import run_with_ngrok
from werkzeug.utils import secure_filename
import face_recognition
import cv2
from threading import Lock

app = Flask(__name__)
run_with_ngrok(app)

# Set the allowed file extensions for image and video files
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif', 'mp4', 'avi', 'mkv'}

# Configure the upload folder and allowed file extensions
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = ALLOWED_EXTENSIONS


# Function to check if a file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


stop_video = False
stop_video_lock = threading.Lock()

polltext = '0 Matches Found '
polltextlive = '0 Matches Found '

pollframe = ''
pollframelive = ''


# Flask route for stopping the video stream
@app.route('/stop_video', methods=['POST'])
def stop_video():
    global stop_video
    print(stop_video)
    with stop_video_lock:
        stop_video = True  # Set the stop_video flag to True to stop the video stream
    return 'Video stream stopped'

# Define a route for polling
@app.route('/poll', methods=['GET'])
def poll():
    # Return a JSON response with the data to be polled
    global polltext
    data = {'message': polltext}
    return jsonify(data)


# Define a route for polling
@app.route('/poll2', methods=['GET'])
def poll2():
    global pollframe
    if pollframe != '':
        data = {'frame': pollframe}
        return jsonify(data)
    else:
        # Handle case where pollframe is empty or not defined
        return jsonify({'error': 'No frame data available'})


# Define a route for polling
@app.route('/polllive', methods=['GET'])
def polllive():
    # Return a JSON response with the data to be polled
    global polltextlive
    data = {'message': polltextlive}
    return jsonify(data)


# Define a route for polling
@app.route('/poll2live', methods=['GET'])
def poll2live():
    global pollframelive
    if pollframelive != '':
        data = {'frame': pollframelive}
        return jsonify(data)
    else:
        # Handle case where pollframe is empty or not defined
        return jsonify({'error': 'No frame data available'})


# Route for the home page
@app.route('/')
def home():
    return render_template('home.html')


@app.route('/detection')
def page2():
    return render_template('detection.html')


@app.route('/viddetect')
def page3():
    return render_template('video_detect.html')


@app.route('/livedetect')
def page5():
    return render_template('live_detect.html')


@app.route('/register')
def page4():
    return render_template('register.html')


# Route for handling file upload and recorded face recognition
@app.route('/upload', methods=['POST'])
def upload():
    # Check if the files are present in the request
    if 'images[]' not in request.files or 'video' not in request.files:
        return 'Error: Images and video file not found in the request'

    # Get the uploaded files
    images = request.files.getlist('images[]')
    video = request.files['video']

    # Process the uploaded images
    known_faces = []
    for image in images:
        if image and allowed_file(image.filename):
            print('running1')
            filename = secure_filename(image.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image.save(image_path)
            image = face_recognition.load_image_file(image_path)
            face_encoding = face_recognition.face_encodings(image)[0]
            known_faces.append(face_encoding)

    # Process the uploaded video
    if video and allowed_file(video.filename):
        print('running')
        filename = secure_filename(video.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        video.save(video_path)

        # Open the video file using OpenCV
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            # print('running3')
            if not ret:
                break
            # Detect faces in the video frame
            face_locations = face_recognition.face_locations(frame)
            face_encodings = face_recognition.face_encodings(
                frame, face_locations)

            for face_location, face_encoding in zip(face_locations, face_encodings):
                # Compare the detected face with known faces
                matches = face_recognition.compare_faces(
                    known_faces, face_encoding)

                # Calculate the percentage of face match
                face_match_percentage = matches.count(
                    True) / len(matches) * 100

                # Draw a rectangle around the detected face
                top, right, bottom, left = face_location
                cv2.rectangle(frame, (left, top),
                              (right, bottom), (0, 255, 0), 2)

                # Display the percentage of face match under the detected face
                cv2.putText(frame, f"Match: {face_match_percentage:.2f}%", (left, bottom + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Display the processed frame
                print(face_match_percentage)
                if face_match_percentage > 98.0:
                    global polltext
                    polltext = 'Person found with match percentage ' + str(face_match_percentage)
                    print('Match found')
                    # Save the frame as an image
                    frame_filename = f"frame_{top}_{right}_{bottom}_{left}.jpg"
                    frame_path = os.path.join(app.config['UPLOAD_FOLDER'], frame_filename)
                    cv2.imwrite(frame_path, frame)

                _, buffer = cv2.imencode('.jpg', frame)
                global pollframe
                pollframe = base64.b64encode(buffer).decode('utf-8')
                cv2.waitKey(1) == ord('q')  # Add a delay to allow time for display
        # return render_template('index.html', frame=frame_base64)
        cap.release()
        cv2.destroyAllWindows()
    return 'Face recognition complete'


# Route for handling file upload and recorded face recognition
@app.route('/uploadlive', methods=['POST'])
def uploadlive():
    # Check if the files are present in the request
    if 'images[]' not in request.files:
        return 'Error: Images and video file not found in the request'

    # Get the uploaded files
    images = request.files.getlist('images[]')

    # Process the uploaded images
    known_faces = []
    for image in images:
        if image and allowed_file(image.filename):
            print('running1')
            filename = secure_filename(image.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image.save(image_path)
            image = face_recognition.load_image_file(image_path)
            face_encoding = face_recognition.face_encodings(image)[0]
            known_faces.append(face_encoding)

        # Open the video file using OpenCV
        cap = cv2.VideoCapture(-1)
        while cap.isOpened():
            ret, frame = cap.read()
            print('accesing')
            global stop_video
            print(stop_video)
            # with stop_video_lock:
            #     if stop_video:
            #         break
            # print('running3')
            if not ret:
                break
            # Detect faces in the video frame
            face_locations = face_recognition.face_locations(frame)
            face_encodings = face_recognition.face_encodings(
                frame, face_locations)

            for face_location, face_encoding in zip(face_locations, face_encodings):
                # Compare the detected face with known faces
                matches = face_recognition.compare_faces(
                    known_faces, face_encoding)

                # Calculate the percentage of face match
                face_match_percentage = matches.count(
                    True) / len(matches) * 100

                # Draw a rectangle around the detected face
                top, right, bottom, left = face_location
                cv2.rectangle(frame, (left, top),
                              (right, bottom), (0, 255, 0), 2)

                # Display the percentage of face match under the detected face
                cv2.putText(frame, f"Match: {face_match_percentage:.2f}%", (left, bottom + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Display the processed frame
                print(face_match_percentage)
                if face_match_percentage > 98.0:
                    global polltextlive
                    polltextlive = 'Person found with match percentage ' + str(face_match_percentage)
                    print('Match found')
                    # Save the frame as an image
                    frame_filename = f"frame_{top}_{right}_{bottom}_{left}.jpg"
                    frame_path = os.path.join(app.config['UPLOAD_FOLDER'], frame_filename)
                    cv2.imwrite(frame_path, frame)

                _, buffer = cv2.imencode('.jpg', frame)
                global pollframelive
                pollframelive = base64.b64encode(buffer).decode('utf-8')
                cv2.waitKey(1)  # Add a delay to allow time for display
        # return render_template('index.html', frame=frame_base64)
        cap.release()
        cv2.destroyAllWindows()
    return 'Face recognition complete'


if __name__ == '__main__':
    app.run()
