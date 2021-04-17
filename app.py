# Import the flask and opencv libraries
from flask import Flask, render_template, Response, request
from flask_wtf.csrf import CSRFProtect
import cv2

from opencv_examples import face_and_eye_detection, calibrate_camera, detect_aruco_markers, num_images_calibrated, calibrate_saved_images

# Initialize the Flask app
app = Flask(__name__)

# Exempt csrf for form submissions so we don't have to reload the page every time we click a button
csrf = CSRFProtect(app)

# Flags for enabling which button the user clicked
b_face_and_eye, b_calibrate, b_aruco_markers, b_take_calibration_image = (False,)*4

# Render our index page
@app.route('/')
def index():
    return render_template('index.html', num_calibrations=num_images_calibrated)

# Enable opencv to capture our webcam
capture = cv2.VideoCapture(0)
# Video capture also lets you specify video files, image sequences, and additional cameras

# Here, we're going to define a simple function that reads our camera image and sends it from the server to client
def create_frames():
    global b_take_calibration_image

    while True:
        # Grab the current camera image
        success, frame = capture.read()

        if success:
            ############# Call Examples ##############
            if b_face_and_eye:
                frame = face_and_eye_detection(frame)
            elif b_calibrate:
                frame = calibrate_camera(frame, b_take_calibration_image)
                b_take_calibration_image = False # reset each time
            elif b_aruco_markers:
                frame = detect_aruco_markers(frame)
            ##########################################

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # We use yield here to produce a sequence of frames
            # We also retain our while loop's previous state to resume where it left off
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n') 
                    ## Server -> Client ##
                    # Formats the frame as a response chunk with a content type of image/jpeg
        else:
            break

# Return the stream of images to be displayed
@app.route('/video_feed')
def video_feed():
    return Response(create_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

## Handles form logic to see if the user is on the calibration page or not ##
@csrf.exempt
@app.route('/form_calibration', methods=['GET'])
def form_calibration():
    return render_template('calibrationform.html', num_calibrations=num_images_calibrated)

@csrf.exempt
@app.route('/form_main', methods=['GET'])
def form_main():
    return render_template('mainform.html')
##############################################################################

# Get the name of the button that the user clicked
@csrf.exempt
@app.route('/handle_data', methods=['POST'])
def handle_data():
    clickedName = request.json['data']
    
    # Reset our active flags when clicking a form button
    global b_face_and_eye, b_aruco_markers, b_take_calibration_image, b_calibrate
    b_face_and_eye, b_aruco_markers, b_take_calibration_image = (False,)*3

    if b_calibrate:
        if clickedName == 'capture':
            b_take_calibration_image = True
        elif clickedName == 'calibrate': # Calibrate saved images
            calibrate_saved_images()
        elif clickedName == 'finish':
            b_calibrate = False
    else:
        if clickedName == 'face-eye':
            b_face_and_eye = True
        elif clickedName == 'calibrate':
            b_calibrate = True
        elif clickedName == 'aruco':
            b_aruco_markers = True

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=False) # Run our app on localhost:5000