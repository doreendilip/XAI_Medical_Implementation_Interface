from flask import Flask, request, render_template, redirect, url_for
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model, Model
import numpy as np
import cv2
import tensorflow as tf
from skimage.transform import resize

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'  # Folder to store uploaded images

# Load your trained model
model_path = 'models/Bmodel.keras'
model = load_model(model_path)
print(f"Model loaded from: {model_path}")

# Class labels (adjust according to your model's classes)
label_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
print(f"Upload folder set to: {app.config['UPLOAD_FOLDER']}")

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle image upload and prediction
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        print("No file part in the request.")
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        print("No file selected.")
        return redirect(request.url)

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print(f"Saving uploaded file to: {filepath}")
        file.save(filepath)

        # Run predictions and Grad-CAM generation
        pred_label, gradcam_image_path = run_gradcam(filepath)

        # Pass only the filename to result.html, not the full path
        return render_template(
            'result.html',
            image_path=os.path.basename(filepath),
            prediction=pred_label,
            gradcam_path=os.path.basename(gradcam_image_path)
        )

# Prediction and Grad-CAM function
def run_gradcam(filepath):
    print(f"Running Grad-CAM for image at: {filepath}")
    
    # Load and preprocess the image
    img = cv2.imread(filepath)
    img_resized = cv2.resize(img, (224, 224))
    img_array = np.expand_dims(img_resized / 255.0, axis=0)
    
    # Make a prediction
    predictions = model.predict(img_array)
    pred_index = np.argmax(predictions[0])
    pred_label = label_names[pred_index]
    print(f"Predicted label: {pred_label}")

    # Generate Grad-CAM
    gradcam_image_path = generate_gradcam(img_array, pred_index, filepath, pred_label)

    return pred_label, gradcam_image_path

# Grad-CAM generation function
def generate_gradcam(img_array, pred_index, filepath, pred_label):
    print(f"Generating Grad-CAM for predicted label: {pred_label}")
    
    # Access the inner Xception model
    xception_base = model.get_layer('xception')
    last_conv_layer_name = 'block14_sepconv2_act'
    conv_layer_output = xception_base.get_layer(last_conv_layer_name).output
    model_output = xception_base.output
    grad_model = Model(inputs=xception_base.input, outputs=[conv_layer_output, model_output])

    # Compute gradients
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    # Compute heatmap
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)  # Normalize heatmap between 0 and 1

    # Resize heatmap and superimpose it on the original image
    img = cv2.imread(filepath)
    if img is None:
        print(f"Error: Could not read image at {filepath}")
    img = cv2.resize(img, (224, 224))
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)  # No inversion, so red represents high values
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(heatmap, 0.4, img, 0.6, 0)

    # Save the Grad-CAM image
    gradcam_path = os.path.join(app.config['UPLOAD_FOLDER'], 'gradcam.png')
    cv2.imwrite(gradcam_path, superimposed_img)
    print(f"Grad-CAM image saved at: {gradcam_path}")

    return gradcam_path

if __name__ == '__main__':
    app.run(debug=True)
