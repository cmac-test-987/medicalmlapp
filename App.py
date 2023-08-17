import streamlit as st
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
import numpy as np

@st.cache_resource
def load_my_model():
    return load_model('MedicalML_ResNet.h5')

model = load_my_model()

# Streamlit web app
st.title("X-Ray Image Classifier")

col1, col2, col3 = st.columns([1,6,1])

with col2:
    uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png", "webp"])

    if uploaded_file is not None:
        image = load_img(uploaded_file, target_size=(224, 224))
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")

        img_array = img_to_array(image)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)

        confidence_for_Normal_XRay = predictions[0][0]

        if confidence_for_Normal_XRay >= .5:
            predicted_class = "Pneumonia_XRay"
            confidence = confidence_for_Normal_XRay
        else:
            predicted_class = "Normal_XRay"
            confidence = 1 - confidence_for_Normal_XRay

        st.write(f"Predicted Class: {predicted_class} with {confidence * 100:.2f}% confidence")

# Feedback System
st.write("## We value your feedback!")
feedback = st.slider("Rate your experience (1-10)", 1, 10)
if st.button("Submit Feedback"):
    st.write(f"Thank you for your feedback! You rated the app as {feedback}/10")
    # Optionally, you can save this feedback to a database or file
