import streamlit as st
import numpy as np
import cv2
from keras.models import load_model
from keras.preprocessing import image

model = load_model('mymodel.h5')

st.title("😷 Face Mask Detector")
st.write("Upload an image to check if the person is wearing a mask or not.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    st.image(img, caption='Uploaded Image', use_column_width=True)

    img_resized = cv2.resize(img, (150, 150))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0 

    prediction = model.predict(img_array)

    if prediction[0][0] > 0.5:
        st.error("🚫 No Mask Detected!")
    else:
        st.success("✅ Mask Detected!")

st.write("---")
st.caption("Developed by Prem Kumar 🚀")
