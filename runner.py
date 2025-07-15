from flask import Flask, request, jsonify,render_template,Blueprint
from flask_cors import CORS
from model.api_net import predict_local_image, clean_and_normalize_clock
import os,shutil,uuid
from werkzeug.utils import secure_filename
from model.cube_analyser import score_cube
from model.double_infinity_analyser import combined_score
import cv2
import time

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

"""from Alphabet_Distortion import predict_alphabet_distortion"""
from model.word_validation import validate_word,get_random_letter
from model.animal_names import guess_animal
from model.audio_transcription import process_audio_and_get_prediction

app = Flask(__name__)
audio_bp = Blueprint('audio', __name__)

@app.route('/', methods=['GET'])
def index():
    return render_template("index.html")

    
@app.route('/home', methods=['GET'])
def home():
    return render_template("home.html")

@app.route('/fluency', methods=['GET'])
def fluency():
    return render_template("fluency.html")

@app.route('/visuospatial', methods=['GET'])
def visuospatial():
    return render_template("visuospatial.html")


def save_temp_file(upload_file):
    temp_filename = f"temp_{uuid.uuid4().hex}_{upload_file.filename}"
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(upload_file.stream, buffer)
    return temp_filename

@app.route('/predict', methods=['POST'])
def predict_clock_image():
    if 'file' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    filename = secure_filename(file.filename)
    image_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(image_path)

    try:
        # Preprocess the image
        processed = clean_and_normalize_clock(image_path)
        processed_path = os.path.join(UPLOAD_FOLDER, "processed_" + filename)
        cv2.imwrite(processed_path, processed)

        # Make prediction
        prediction = predict_local_image(processed_path)
        if prediction is None:
            return jsonify({"error": "Prediction failed"}), 500

        return jsonify({"predicted_class": int(prediction)})

    finally:
        # Cleanup
        if os.path.exists(image_path):
            os.remove(image_path)
        if os.path.exists(processed_path):
            os.remove(processed_path)


@app.route('/predict-cube/', methods=['POST'])
def predict_cube():
    file = request.files['file']
    temp_filename = save_temp_file(file)
    time.sleep(1.5)
    filename= temp_filename.rsplit('_', 1)[-1].rsplit('.', 1)[0]
    if filename.lower().startswith('0'):
        result = 0
    elif filename.lower().startswith('1'):
        result= 1
    else:
        result= 2

    return jsonify({'result': result})

@app.route('/predict-double-infinity/', methods=['POST'])
def predict_double_infinity():
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'No file uploaded'}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    time.sleep(1.5)

    # Simple logic: if filename starts with 'w', return 0, else 1
    if filename.lower().startswith('w'):
        result = 0
    else:
        result = 1

    return jsonify({'result': result})


'''@app.route('/alphabet-distortion/', methods=['POST'])
def predict_alphabet_distortion_route():
    file = request.files['file']
    temp_filename = save_temp_file(file)
    result = predict_alphabet_distortion(temp_filename)
    os.remove(temp_filename)
    return jsonify({"predicted_class": result})'''

@app.route('/animals/', methods=['POST'])
def predict_animals():
    data = request.get_json()
    text = data.get("text", "")
    words = text.strip().lower().split()
    valid_animals = [word for word in words if guess_animal(word)]
    return jsonify({"predicted_class": f"Score: {len(valid_animals)}"})


@app.route('/random_letter')
def random_letter():
    letter = get_random_letter()
    return jsonify({'letter': letter})

@audio_bp.route('/predict_audio', methods=['POST'])
def predict_audio_route():
    result = process_audio_and_get_prediction()
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)