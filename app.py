import base64
from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import io
# Import Spotify credentials from config.py
from config import SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET
import requests

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Use a non-interactive backend for Matplotlib
import matplotlib
matplotlib.use('Agg')

script_dir = os.path.dirname(os.path.abspath(__file__))
# Change the working directory to the script's directory
os.chdir(script_dir)
# Load the ResNet50V2 model
ResNet50V2_Model = tf.keras.models.load_model('resnet.h5')

# Load the dataset
Music_Player = pd.read_csv('data_moods2.csv')

# Define the Flask app
app = Flask(__name__, static_url_path='/static')

# Class names for the prediction
class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Get Spotify access token
def get_spotify_token():
    auth_url = 'https://accounts.spotify.com/api/token'
    auth_header = {
        'Authorization': 'Basic ' + base64.b64encode((SPOTIFY_CLIENT_ID + ':' + SPOTIFY_CLIENT_SECRET).encode()).decode('utf-8'),
    }
    auth_data = {
        'grant_type': 'client_credentials',
    }
    response = requests.post(auth_url, headers=auth_header, data=auth_data)
    response_data = response.json()
    return response_data['access_token']

# Function to search for songs on Spotify
def search_spotify(song_name, token):
    search_url = 'https://api.spotify.com/v1/search'
    headers = {
        'Authorization': f'Bearer {token}',
    }
    params = {
        'q': song_name,
        'type': 'track',
        'limit': 1,
    }
    response = requests.get(search_url, headers=headers, params=params)
    response_data = response.json()
    if response_data['tracks']['items']:
        return response_data['tracks']['items'][0]['external_urls']['spotify']
    else:
        return None


# Function to recommend songs
def Recommend_Songs(pred_class, genre):
    if pred_class in ['Happy', 'Sad']:
        Play = Music_Player[Music_Player['mood'] == 'Happy']
    elif pred_class in ['Fear', 'Angry', 'Disgust']:
        Play = Music_Player[Music_Player['mood'] == 'Calm']
    elif pred_class in ['Surprise', 'Neutral']:
        Play = Music_Player[Music_Player['mood'] == 'Energetic']
    
    Play = Play.sort_values(by="popularity", ascending=False)
    if genre == '' or genre is None:
        Play = Play[:5].reset_index(drop=True)
        return Play['name'].tolist()
    mask = pd.notna(Play['genres'])
    filtered_data = Play[mask]
    filtered_data = filtered_data[filtered_data['genres'].str.contains(genre)]
    filtered_data = filtered_data[:5].reset_index(drop=True)
    return filtered_data['name'].tolist()

# Function to load and prepare the image
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def load_and_prep_image(image_stream, img_shape=224):
    image_bytes = image_stream.read()
    
    # Convert the image bytes to a NumPy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    # Decode the image array
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        print("Failed to decode image")
        return None, None
    
    GrayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(GrayImg, 1.1, 4)

    if len(faces) == 0:
        print("No faces detected")
        return None, None

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

        # Ensure there's only one detected face to return
        if len(faces) == 1:
            RGBImg = cv2.cvtColor(roi_Img, cv2.COLOR_BGR2RGB)
            RGBImg = cv2.resize(RGBImg, (img_shape, img_shape))
            RGBImg = RGBImg / 255.
            return RGBImg, img_base64

    print("Multiple faces detected")
    return None, None

# Function to predict and recommend songs
def pred_and_recommend(image_file, class_names, genre):
    img, img_base64 = load_and_prep_image(image_file)
    if img is None:
        print("Image preprocessing failed")
        return None, None, None
    pred = ResNet50V2_Model.predict(np.expand_dims(img, axis=0))
    pred_class = class_names[pred.argmax()]
    songs = Recommend_Songs(pred_class, genre)
    # Get Spotify access token
    token = get_spotify_token()
    song_links = []
    for song in songs:
        link = search_spotify(song, token)
        song_links.append({'title': song, 'link': link})
    return pred_class, song_links, img_base64 

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    image_file = request.files['file']
    genre = request.form.get('genre')
    
    pred_class, songs, img_base64 = pred_and_recommend(image_file.stream, class_names, genre)
    if pred_class is None:
        return jsonify({'class': None, 'songs': [], 'image': None})

    return jsonify({'class': pred_class, 'songs': songs, 'image': img_base64})

if __name__ == '__main__':
    app.run(debug=True)
