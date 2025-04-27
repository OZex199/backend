from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import pytesseract
from PIL import Image
import os

app = Flask(__name__)

# Configuration for file uploads
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/upload', methods=['POST'])
def upload_receipt():
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

            return jsonify({'message': 'OCR processed successfully', 'extracted_text': text}), 200
        except Exception as e:
            return jsonify({'message': str(e)}), 500

    return jsonify({'message': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
