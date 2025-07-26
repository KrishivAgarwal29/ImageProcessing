import os
import cv2
import numpy as np
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Needed for flash
app.config['UPLOAD_FOLDER'] = 'static'  # Where to save output images

@app.route('/')
def upload_form():
    return render_template('upload.html')


@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    image_file = request.files['file']
    if image_file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    operation_selection = request.form.get('image_type_selection')
    filename = secure_filename(image_file.filename)

    reading_file_data = image_file.read()
    image_array = np.frombuffer(reading_file_data, dtype='uint8')
    decode_array_to_img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if decode_array_to_img is None:
        flash('Image decoding failed.')
        return redirect(request.url)

    # Apply selected operation
    if operation_selection == 'gray':
        file_data = make_grayscale(decode_array_to_img)
    elif operation_selection == 'sketch':
        file_data = image_sketch(decode_array_to_img)
    elif operation_selection == 'oil':
        file_data = oil_effect(decode_array_to_img)
    elif operation_selection == 'rgb':
        file_data = rgb_effect(decode_array_to_img)
    elif operation_selection == 'water':
        file_data = water_color_effect(decode_array_to_img)
    elif operation_selection == 'invert':
        file_data = invert(decode_array_to_img)
    elif operation_selection == 'hdr':
        file_data = HDR(decode_array_to_img)
    else:
        flash('Invalid operation selected')
        return redirect(request.url)

    output_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    with open(output_path, 'wb') as f:
        f.write(file_data)

    return render_template('upload.html', filename=filename)


# --- Image Processing Functions ---

def make_grayscale(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, output = cv2.imencode('.png', gray)
    return output.tobytes()

def image_sketch(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inverted = cv2.bitwise_not(gray)
    blurred = cv2.GaussianBlur(inverted, (111, 111), 0)
    sketch = cv2.divide(gray, cv2.bitwise_not(blurred), scale=256.0)
    _, output = cv2.imencode('.png', sketch)
    return output.tobytes()

def oil_effect(img):
    filtered = cv2.bilateralFilter(img, 9, 75, 75)
    stylized = cv2.stylization(filtered, sigma_s=60, sigma_r=0.07)
    _, output = cv2.imencode('.png', stylized)
    return output.tobytes()

def rgb_effect(img):
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    _, output = cv2.imencode('.png', rgb_img)
    return output.tobytes()

def water_color_effect(img):
    water = cv2.stylization(img, sigma_s=60, sigma_r=0.6)
    _, output = cv2.imencode('.png', water)
    return output.tobytes()

def invert(img):
    inverted = cv2.bitwise_not(img)
    _, output = cv2.imencode('.png', inverted)
    return output.tobytes()

def HDR(img):
    hdr = cv2.detailEnhance(img, sigma_s=12, sigma_r=0.15)
    _, output = cv2.imencode('.png', hdr)
    return output.tobytes()

@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename=filename))


if __name__ == "__main__":
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
