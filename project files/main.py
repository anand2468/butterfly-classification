from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import json


model = load_model('./vgg16_butterfly_model.keras')

# Define preprocessing (adjust to your model's expected input)
def preprocess_image(image, target_size=(224, 224)):
    image = image.resize(target_size)
    image_array = np.array(image)

    if image_array.shape[-1] == 4:
        image_array = image_array[..., :3]  # remove alpha channel if present

    image_array = image_array / 255.0  # normalize
    image_array = np.expand_dims(image_array, axis=0)  # batch dimension
    return image_array



app = Flask(__name__)

with open('class_indices.json') as f:
    class_indices = json.load(f)

index_to_label = {v: k for k, v in class_indices.items()}

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        image = Image.open(file.stream).convert('RGB')
        img = preprocess_image(image)
        prediction = model.predict(img)
        prediction_index = np.argmax(prediction, axis=1) 

        data =  {'label': index_to_label[prediction_index[0]], 'confidence': float(prediction[0][prediction_index[0]])* 100}
        if data['confidence'] < 65:
            data['label'] = 'Unknown or no butterfly detected'
            data['confidence'] = 0.0
        return render_template('result.html', data=data)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)