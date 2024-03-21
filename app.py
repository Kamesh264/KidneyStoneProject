import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2

# Load the trained model
model = tf.keras.models.load_model('model_dl.h5')

# Define the prediction function
def predict(image):
    # Resize and preprocess the image
    image_resized = cv2.resize(image, (200, 200))
    gray_image = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    gray_image = np.expand_dims(gray_image, axis=-1)
    gray_image = gray_image / 255.0

    # Perform inference on the preprocessed image
    predictions = model.predict(np.expand_dims(gray_image, axis=0))
    return predictions


# Streamlit app
def main():
    st.title('Kideny Stone Prediction')

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        # Display the uploaded image
        image = np.array(Image.open(uploaded_file),)
        st.image(image, caption='Uploaded Image', use_column_width=100)

        # Make prediction
        if st.button('Predict'):
            prediction = predict(image)
            obj_list = ['Normal', 'Tumor', 'Cyst', 'Stone']
            predicted_label = obj_list[np.argmax(prediction)]
            st.write('Prediction:', predicted_label)

if __name__ == '__main__':
    main()
