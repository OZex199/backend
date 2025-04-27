from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import pytesseract
from PIL import Image
import os
import requests
from functools import wraps
import jwt

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)

# Configuration for file uploads
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your_secret_key')

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Decorator to verify JWT token
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')

        if not token:
            return jsonify({'message': 'Token is missing!'}), 403

        try:
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
            current_user = data['username']
        except Exception as e:
            return jsonify({'message': 'Token is invalid!'}), 403

        return f(current_user, *args, **kwargs)
    return decorated

@app.route('/upload', methods=['POST'])
@token_required
def upload_receipt(current_user):
    if 'file' not in request.files:
        return jsonify({'message': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'message': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # OCR processing on the image file
        try:
            img = Image.open(filepath)
            text = pytesseract.image_to_string(img)

            # Send the extracted text to the Rust microservice for further processing
            rust_url = 'http://127.0.0.1:8080/api/process'
            response = requests.post(rust_url, json={'text': text})

            if response.status_code == 200:
                return jsonify({'message': 'Receipt processed successfully', 'data': response.json()}), 200
            else:
                return jsonify({'message': 'Failed to process receipt'}), 500
        except Exception as e:
            return jsonify({'message': str(e)}), 500

    return jsonify({'message': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
