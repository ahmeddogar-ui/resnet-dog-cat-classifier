# routes/DOGCAT.py

from flask import Blueprint, request, jsonify
from services.predictor import predict
import os

predict_bp = Blueprint('predict', __name__)


@predict_bp.route('/predict', methods=['POST'])
def predict_endpoint():
    if 'file' not in request.files:
        return jsonify({'error': 'no file uploaded'}), 400

    file = request.files['file']

    # Check if file is empty
    if file.filename == '':
        return jsonify({'error': 'empty filename'}), 400

    # Check if file is an image
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in allowed_extensions:
        return jsonify({'error': f'invalid file type: {file_ext}. Use jpg, png, or gif'}), 400

    # Save with original extension
    img_path = f'temp_image{file_ext}'
    file.save(img_path)

    try:
        result = predict(img_path)
        os.remove(img_path)
        return jsonify({'prediction': result})
    except Exception as e:
        # Clean up temp file on error
        if os.path.exists(img_path):
            os.remove(img_path)
        return jsonify({'error': str(e)}), 500
