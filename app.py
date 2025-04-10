from flask import Flask, flash, request, redirect, url_for, render_template
import urllib.request
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input

# Loading Models
covid_model = load_model('models/covid.h5')
braintumor_model = load_model('models/braintumor.h5')
alzheimer_model = load_model('models/alzheimer_model.h5')
pneumonia_model = load_model('models/pneumonia_model.h5')

# Configuring Flask
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "secret key"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('homepage.html')

@app.route('/covid')
def covid():
    return render_template('covid.html')

@app.route('/braintumor')
def brain_tumor():
    return render_template('braintumor.html')

@app.route('/alzheimer')
def alzheimer():
    return render_template('alzheimer.html')

@app.route('/pneumonia')
def pneumonia():
    return render_template('pneumonia.html')

@app.route('/resultc', methods=['POST'])
def resultc():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            img = cv2.imread('static/uploads/'+filename)
            img = cv2.resize(img, (224, 224))
            img = img.reshape(1, 224, 224, 3) / 255.0
            pred = (covid_model.predict(img) >= 0.5).astype(int)[0, 0]
            return render_template('resultc.html', filename=filename, r=pred)
        flash('Allowed image types are - png, jpg, jpeg')
        return redirect(request.url)

@app.route('/resultbt', methods=['POST'])
def resultbt():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            img = cv2.imread('static/uploads/'+filename)
            img = cv2.resize(img, (224, 224))
            img = preprocess_input(img.reshape(1, 224, 224, 3))
            pred = (braintumor_model.predict(img) >= 0.5).astype(int)[0, 0]
            return render_template('resultbt.html', filename=filename, r=pred)
        flash('Allowed image types are - png, jpg, jpeg')
        return redirect(request.url)

@app.route('/resulta', methods=['POST'])
def resulta():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            img = cv2.imread('static/uploads/'+filename)
            img = cv2.resize(img, (176, 176))
            img = img.reshape(1, 176, 176, 3) / 255.0
            pred = alzheimer_model.predict(img)[0].argmax()
            return render_template('resulta.html', filename=filename, r=pred)
        flash('Allowed image types are - png, jpg, jpeg')
        return redirect('/')

@app.route('/resultp', methods=['POST'])
def resultp():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            img = cv2.imread('static/uploads/'+filename)
            img = cv2.resize(img, (150, 150))
            img = img.reshape(1, 150, 150, 3) / 255.0
            pred = (pneumonia_model.predict(img) >= 0.5).astype(int)[0, 0]
            return render_template('resultp.html', filename=filename, r=pred)
        flash('Allowed image types are - png, jpg, jpeg')
        return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True)
