from flask import Flask, request, render_template, redirect, url_for
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
from tensorflow.keras.layers import Layer



app = Flask(__name__)




# Load the model with the custom layer
model = load_model('D:\Dog&cat\Model_catDogs.h5')


def predict_image(img_path):
    # Load image
    image = Image.open(img_path)
    image = image.resize((150, 170))  # Resize the image to the size your model expects
    image = img_to_array(image) / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(image)
    return "Dog" if prediction[0][1] > 0.5 else "Cat"

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            file_path = 'static/' + file.filename
            file.save(file_path)

            # Make prediction
            prediction = predict_image(file_path)

            return render_template('result.html', prediction=prediction, image_path=file_path)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
