import streamlit as st
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import sqlite3
import pandas as pd

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

# Create a new SQLite database or connect to existing one
con = sqlite3.connect('feedback.db')
cursor = con.cursor()

# Create a table for feedback if not already present
cursor.execute('''
    CREATE TABLE IF NOT EXISTS feedback_data (
        feedback INTEGER
    )
''')
con.commit()

# Save feedback to the SQLite database
def save_feedback(feedback):
    cursor.execute("INSERT INTO feedback_data (feedback) VALUES (?)", (feedback,))
    con.commit()

# Feedback System
st.write("## We value your feedback!")
feedback1 = st.slider("Rate your experience (1-10)", 1, 10, key="feedback_slider1")

if st.button("Submit Feedback"):
    save_feedback(feedback1)
    st.write(f"Thank you for your feedback! You rated the app as {feedback1}/10")

# Display feedback summary
if st.button("Show Feedback Summary"):
    # Fetch feedback data from database
    cursor.execute("SELECT feedback, COUNT(*) as count FROM feedback_data GROUP BY feedback")
    feedback_data = cursor.fetchall()

    # Convert the feedback data to DataFrame for visualization
    df = pd.DataFrame(feedback_data, columns=["Feedback", "Count"])
    st.bar_chart(df.set_index("Feedback"))
