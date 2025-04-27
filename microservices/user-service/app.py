from flask import Flask, request, jsonify, send_from_directory
from flask_mysqldb import MySQL
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import os
import jwt
import datetime
import requests
import pytesseract
from PIL import Image, ImageFilter
import fitz
import logging
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
from flask_cors import CORS
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
CORS(app)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your_secret_key')
app.config['MYSQL_HOST'] = os.getenv('MYSQL_HOST')
app.config['MYSQL_USER'] = os.getenv('MYSQL_USER')
app.config['MYSQL_PASSWORD'] = os.getenv('MYSQL_PASSWORD')
app.config['MYSQL_DB'] = os.getenv('MYSQL_DATABASE')
app.config['MYSQL_PORT'] = int(os.getenv('MYSQL_PORT'))
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'pdf'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
mysql = MySQL(app)

logging.basicConfig(level=logging.INFO)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'error': 'Missing token'}), 403
        try:
            token = token.split()[1]
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
            current_user = data['username']
        except Exception:
            return jsonify({'error': 'Invalid token'}), 403
        return f(current_user, *args, **kwargs)
    return decorated

def extract_text_from_pdf(pdf_path):
    text = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        text += page.get_text()
    return text

def preprocess_image(filepath):
    img = Image.open(filepath)
    img = img.convert('L')
    img = img.filter(ImageFilter.SHARPEN)
    return img
def ensure_categories_table_exists():
    cur = mysql.connection.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS categories (
            id INT AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            user VARCHAR(255) NOT NULL
        )
    """)
    mysql.connection.commit()
    cur.close()

def save_receipt_to_db(user, receipt_data, filename):
    cur = mysql.connection.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS receipts (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user VARCHAR(255),
            merchant VARCHAR(255),
            date DATE,
            subtotal FLOAT,
            tax FLOAT,
            total FLOAT,
            items JSON,
            file_path VARCHAR(255)
        )
    """)
    mysql.connection.commit()

    cur.execute("""
        INSERT INTO receipts (user, merchant, date, subtotal, tax, total, items, file_path)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        user,
        receipt_data['merchant'],
        receipt_data['date'],
        receipt_data['subtotal'],
        receipt_data['tax'],
        receipt_data['total'],
        json.dumps(receipt_data['items']),
        filename
    ))
    mysql.connection.commit()
    cur.close()

@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    username, password = data.get('username'), data.get('password')
    if not username or not password:
        return jsonify({'error': 'Username and password required'}), 400
    hashed_pw = generate_password_hash(password)
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM users WHERE username=%s", (username,))
    if cur.fetchone():
        return jsonify({'error': 'Username already exists'}), 409
    cur.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, hashed_pw))
    mysql.connection.commit()
    cur.close()
    return jsonify({'message': 'Registered successfully'}), 201

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username, password = data.get('username'), data.get('password')
    cur = mysql.connection.cursor()
    cur.execute("SELECT password FROM users WHERE username=%s", (username,))
    result = cur.fetchone()
    cur.close()
    if result and check_password_hash(result[0], password):
        token = jwt.encode({'username': username, 'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=2)}, app.config['SECRET_KEY'])
        return jsonify({'token': token})
    return jsonify({'error': 'Invalid credentials'}), 401

@app.route('/upload', methods=['POST'])
@token_required
def upload_receipt(current_user):
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        if filename.lower().endswith('.pdf'):
            text = extract_text_from_pdf(filepath)
        else:
            img = preprocess_image(filepath)
            text = pytesseract.image_to_string(img, config="--psm 4")

        rust_response = requests.post('http://localhost:8080/api/process', json={'text': text, 'user': current_user})

        if rust_response.status_code == 200:
            receipt_data = rust_response.json()
            save_receipt_to_db(current_user, receipt_data, filename)
            return jsonify({'status': 'Receipt processed', 'receipt': receipt_data})
        else:
            return jsonify({'error': 'Rust service error'}), 500
    except Exception as e:
        logging.error(f"Upload error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/analytics', methods=['GET'])
@token_required
def analytics(current_user):
    cur = mysql.connection.cursor()
    cur.execute("SELECT items FROM receipts WHERE user = %s", (current_user,))
    receipts = cur.fetchall()
    cur.close()

    category_totals = {}
    for (items_json,) in receipts:
        items = json.loads(items_json)
        for item in items:
            cat = item.get('category', 'others')
            category_totals[cat] = category_totals.get(cat, 0) + float(item.get('price', 0))

    return jsonify({
        'monthly_spending': category_totals,
        'potential_savings': sum(category_totals.values()) * 0.1,
        'achieved_savings': sum(category_totals.values()) * 0.03
    })

@app.route('/recommendations', methods=['GET'])
@token_required
def recommendations(current_user):
    try:
        cur = mysql.connection.cursor()
        cur.execute("SELECT items FROM receipts WHERE user = %s", (current_user,))
        receipts = cur.fetchall()
        cur.close()

        corpus = []
        for (items_json,) in receipts:
            items = json.loads(items_json)
            for item in items:
                corpus.append(item.get('name', '') + " " + item.get('category', ''))

        if not corpus:
            return jsonify({'recommendations': []})

        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform(corpus)

        n_clusters = min(3, len(corpus))
        model = KMeans(n_clusters=n_clusters, random_state=42)
        model.fit(X)

        cluster_keywords = np.argsort(model.cluster_centers_, axis=1)[:, -1]
        feature_names = vectorizer.get_feature_names_out()

        recommendations = []
        for idx in cluster_keywords:
            keyword = feature_names[idx]
            recommendations.append({
                "merchant": f"PartnerStore-{keyword.capitalize()}",
                "potential_savings": np.random.randint(100, 1000),
                "distance": f"{round(np.random.uniform(0.5, 5.0), 1)}km"
            })

        return jsonify({'recommendations': recommendations})

    except Exception as e:
        logging.error(f"Recommendation error: {e}")
        return jsonify({'error': 'Recommendation engine failure'}), 500

@app.route('/history', methods=['GET'])
@token_required
def history(current_user):
    cur = mysql.connection.cursor()
    cur.execute("SELECT merchant, date, file_path FROM receipts WHERE user = %s", (current_user,))
    records = cur.fetchall()
    cur.close()

    history = []
    for merchant, date, file_path in records:
        history.append({
            'merchant': merchant,
            'date': str(date),
            'receipt_image_url': f"/uploads/{file_path}"
        })
    return jsonify(history)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/partners', methods=['GET'])
@token_required
def get_partners(current_user):
    try:
        skbt_response = requests.get('https://api.sovcombank.ru/halva/partners', timeout=5)

        if skbt_response.status_code == 200:
            partners_data = skbt_response.json()
            return jsonify({'partners': partners_data, 'source': 'skbt'})

    except Exception as e:
        logging.error(f"Partner API error: {e}")

    fallback_partners = [
        {"name": "PartnerStore-Electronics", "category": "Electronics", "discount": "10%"},
        {"name": "PartnerStore-Fashion", "category": "Clothing", "discount": "5%"},
        {"name": "PartnerStore-Grocery", "category": "Groceries", "discount": "8%"},
        {"name": "PartnerStore-Sports", "category": "Sports", "discount": "7%"},
        {"name": "PartnerStore-Beauty", "category": "Beauty", "discount": "6%"}
    ]
    return jsonify({'partners': fallback_partners, 'source': 'fallback'})
@app.route('/categories', methods=['GET'])
@token_required
def get_categories(current_user):
    ensure_categories_table_exists()  
    cur = mysql.connection.cursor()
    cur.execute("SELECT id, name FROM categories WHERE user=%s", (current_user,))
    categories = cur.fetchall()
    cur.close()
    return jsonify([{'id': c[0], 'name': c[1]} for c in categories])

@app.route('/categories', methods=['POST'])
@token_required
def add_category(current_user):
    ensure_categories_table_exists()  
    data = request.get_json()
    name = data.get('name')
    cur = mysql.connection.cursor()
    cur.execute("INSERT INTO categories (name, user) VALUES (%s, %s)", (name, current_user))
    mysql.connection.commit()
    cur.close()
    return jsonify({'message': 'Category added successfully'})

@app.route('/categories/<int:category_id>', methods=['PUT'])
@token_required
def update_category(current_user, category_id):
    ensure_categories_table_exists()   
    data = request.get_json()
    name = data.get('name')
    cur = mysql.connection.cursor()
    cur.execute("UPDATE categories SET name=%s WHERE id=%s AND user=%s", (name, category_id, current_user))
    mysql.connection.commit()
    cur.close()
    return jsonify({'message': 'Category updated successfully'})

@app.route('/categories/<int:category_id>', methods=['DELETE'])
@token_required
def delete_category(current_user, category_id):
    ensure_categories_table_exists()   
    cur = mysql.connection.cursor()
    cur.execute("DELETE FROM categories WHERE id=%s AND user=%s", (category_id, current_user))
    mysql.connection.commit()
    cur.close()
    return jsonify({'message': 'Category deleted successfully'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
