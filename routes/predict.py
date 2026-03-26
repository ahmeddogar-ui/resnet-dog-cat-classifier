from flask import jsonify, Blueprint
from DOGCAT import model, predict

predict_bp = Blueprint('predict', __name__)
@predict_bp.route('/predict', methods=['POST'])
@app.route('/predict', methods=['POST'])
def predict_endpoint():
    file = request.files['file']

    # Save temporarily
    img_path = 'temp_image.jpg'
    file.save(img_path)

    # Call your predict function
    result = predict(model, img_path)

    # Return JSON
    return jsonify({'prediction': result})
    # → {"prediction": "dog"}
