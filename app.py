import base64
from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import io

script_dir = os.path.dirname(os.path.abspath(__file__))
# Change the working directory to the script's directory
os.chdir(script_dir)
# Load the model
ResNet50V2_Model = tf.keras.models.load_model('ResNet50V2_Model.h5')

# Load the dataset
Music_Player = pd.read_csv('data_moods2.csv')

# Define the Flask app
app = Flask(__name__)

# Class names for the prediction
class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Function to recommend songs
def Recommend_Songs(pred_class):
    if pred_class == 'Disgust':
        Play = Music_Player[Music_Player['mood'] == 'Sad']
    elif pred_class in ['Happy', 'Sad']:
        Play = Music_Player[Music_Player['mood'] == 'Happy']
    elif pred_class in ['Fear', 'Angry']:
        Play = Music_Player[Music_Player['mood'] == 'Calm']
    elif pred_class in ['Surprise', 'Neutral']:
        Play = Music_Player[Music_Player['mood'] == 'Energetic']
    
    Play = Play.sort_values(by="popularity", ascending=False)
    Play = Play[:5].reset_index(drop=True)
    return Play['name'].tolist()

# Function to load and prepare the image
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def load_and_prep_image(image_stream, img_shape=224):
    image_bytes = image_stream.read()
    
    # Convert the image bytes to a NumPy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    # Decode the image array
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return None
    #img = cv2.imread(image_file)
    GrayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(GrayImg, 1.1, 4)

    for x, y, w, h in faces:
        roi_GrayImg = GrayImg[y: y + h, x: x + w]
        roi_Img = img[y: y + h, x: x + w]
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        

    # Plot the image with matplotlib and convert it to base64
        fig, ax = plt.subplots()
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB
        ax.axis('off')

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')

        faces = faceCascade.detectMultiScale(roi_Img, 1.1, 4)
        if len(faces) == 0:
            return None
        else:
            for (ex, ey, ew, eh) in faces:
                img = roi_Img[ey: ey + eh, ex: ex + ew] 

    RGBImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    RGBImg = cv2.resize(RGBImg, (img_shape, img_shape))
    RGBImg = RGBImg / 255.
    return RGBImg, img_base64

# Function to predict and recommend songs
def pred_and_recommend(image_file, class_names):
    img, img_base64 = load_and_prep_image(image_file)
    if img is None:
        return None, None
    pred = ResNet50V2_Model.predict(np.expand_dims  (img, axis=0))
    pred_class = class_names[pred.argmax()]
    songs = Recommend_Songs(pred_class)
    # Encode the image to base64 string
    #_, img_encoded = cv2.imencode('.jpg', original_img)
    #img_base64 = base64.b64encode(img_encoded).decode('utf-8')
    return pred_class, songs, img_base64 

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    image_file = request.files['file']
    #filepath = f'{file.filename}'
    #print(filepath)
    #filepath = f'static/{file.filename}'
    #file.save(filepath)
    
    pred_class, songs, img_base64 = pred_and_recommend(image_file.stream, class_names)
        #if pred_class is None:
    #    return jsonify({'songs': []})
    #return jsonify({'songs': songs})
    return jsonify({'class': pred_class, 'songs': songs, 'image': img_base64})

if __name__ == '__main__':
    app.run(debug=True)
