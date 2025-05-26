from flask import Flask,jsonify,request
from flask_cors import CORS
import cv2
# from keras.models import load_model

import numpy as np
import base64
# json_file = open("facialemotionmodel.json", "r")
# model_json = json_file.read()
# json_file.close()
# model = model_from_json(model_json)

# model.load_weights("facialemotionmodel.h5")
import tensorflow as tf
model = tf.keras.models.load_model("facialemotionmodel.h5")
haar_file=cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade=cv2.CascadeClassifier(haar_file)

# def extract_features(image):
#     feature = np.array(image)
#     feature = feature.reshape(1,48,48,1)
#     return feature/255.0

# webcam=cv2.VideoCapture(0)
# labels = {0 : 'angry', 1 : 'disgust', 2 : 'fear', 3 : 'happy', 4 : 'neutral', 5 : 'sad', 6 : 'surprise'}
# app = Flask(__name__)
# CORS(app,origins='*')
# @app.route('/')
# def hello():
#     return 'Hello World !'

# @app.route('/predictemotion',methods = ['POST'])
# def predict_emotion():
#     data = request.get_json()
#     if data is None:
#         return jsonify({"error":"invalid json data"}),400
#     # print(data)
#     image_string = data['image']
#     image_string = image_string.split(',')
#     if len(image_string)==2:
#         image_string = image_string[1]
#     else:
#         image_string = image_string[0]
        
#     image_data = base64.b64decode(image_string)
#     nparr = np.frombuffer(image_data,np.uint8)
#     img = cv2.imdecode(nparr,cv2.IMREAD_COLOR)
#     gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#     faces=face_cascade.detectMultiScale(img,1.3,5)
#     print(faces)
#     for (p,q,r,s) in faces:
#             image = gray[q:q+s,p:p+r]
#             cv2.rectangle(img,(p,q),(p+r,q+s),(255,0,0),2)
#             image = cv2.resize(image,(48,48))
#             im = extract_features(image)
#             pred = model.predict(im)
#             prediction_label = labels[pred.argmax()]
#             print(prediction_label)
#             return jsonify({"emotion":prediction_label,"x1": str(p),"y1":str(q),"x2":str(r),"y2":str(s)}),200
#     # cv2.imshow("image",img)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()
#     return jsonify({"emotion":"face is not detected","x1":str(0),"y1":str(0),"x2":str(0),"y2":str(0)}),200
    

# if __name__ == '__main__':
#     app.run()
#     # app.run(debug=True)

# import streamlit as st
# import cv2
# import numpy as np
# from keras.models import load_model
# from PIL import Image

# # Load the pre-trained model
# # model = load_model('facimodel.h5')
# # face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# # Define emotions
# emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# # Streamlit UI
# st.title("😄 Real-Time Emotion Detection")
# st.write("Upload an image to detect faces and predict emotions.")

# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     # Convert to OpenCV image
#     img = Image.open(uploaded_file)
#     img_array = np.array(img.convert('RGB'))
#     img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

#     # Detect faces
#     faces = face_cascade.detectMultiScale(img_gray, 1.3, 5)

#     for (x, y, w, h) in faces:
#         roi_gray = img_gray[y:y+h, x:x+w]
#         roi_gray = cv2.resize(roi_gray, (48, 48))
#         roi = roi_gray.astype('float') / 255.0
#         roi = np.reshape(roi, (1, 48, 48, 1))
#         prediction = model.predict(roi)
#         emotion = emotion_labels[np.argmax(prediction)]

#         # Draw bounding box and label
#         cv2.rectangle(img_array, (x, y), (x+w, y+h), (255, 0, 0), 2)
#         cv2.putText(img_array, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
#                     1, (0, 255, 0), 2, cv2.LINE_AA)

#     st.image(img_array, caption="Processed Image", channels="RGB")

import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
from PIL import Image

# # Load model and face detector
# model = load_model('model.h5')
# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def detect_emotions(img):
    img_array = np.array(img.convert('RGB'))
    img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    detected_emotions = []

    faces = face_cascade.detectMultiScale(img_gray, 1.3, 5)
    for (x, y, w, h) in faces:
        roi_gray = img_gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi = roi_gray.astype('float') / 255.0
        roi = np.reshape(roi, (1, 48, 48, 1))

        prediction = model.predict(roi)
        emotion = emotion_labels[np.argmax(prediction)]
        detected_emotions.append(emotion)

        # Draw on image
        cv2.rectangle(img_array, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(img_array, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2, cv2.LINE_AA)

    return img_array, detected_emotions

# Streamlit UI
st.title("😄 Real-Time Emotion Detection")

st.write("Choose input method:")

option = st.radio("", ["Upload Image", "Use Webcam"])

if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        result_img, emotions = detect_emotions(img)
        st.image(result_img, channels="RGB", caption="Processed Image")

        if emotions:
            st.success("Detected Emotions:")
            for i, e in enumerate(emotions, 1):
                st.write(f"Face {i}: **{e}**")
        else:
            st.warning("No face detected.")

elif option == "Use Webcam":
    photo = st.camera_input("Take a photo")
    if photo is not None:
        img = Image.open(photo)
        result_img, emotions = detect_emotions(img)
        st.image(result_img, channels="RGB", caption="Processed Image")

        if emotions:
            st.success("Detected Emotions:")
            for i, e in enumerate(emotions, 1):
                st.write(f"Face {i}: **{e}**")
        else:
            st.warning("No face detected.")
