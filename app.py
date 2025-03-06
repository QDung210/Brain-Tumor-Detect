import streamlit as st
import joblib
import numpy as np
import cv2
from PIL import Image

# Load mô hình SVM và danh sách các lớp
svm_model = joblib.load("svm_model.joblib")
classes = joblib.load("classes.joblib")

def preprocess_image(image):
    """Tiền xử lý ảnh trước khi đưa vào mô hình."""
    image = np.array(image.convert("L"))  # Chuyển ảnh sang grayscale
    image = cv2.resize(image, (64, 64))   # Resize về kích thước phù hợp với mô hình
    image = image.flatten() / 255.0        # Chuyển thành vector và chuẩn hóa
    return image.reshape(1, -1)            # Định dạng lại để đưa vào mô hình

st.title("Dự đoán loại khối u bằng SVM")

uploaded_file = st.file_uploader("Chọn một ảnh y khoa", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Ảnh đã tải lên", use_column_width=True)
    
    if st.button("Dự đoán"):
        processed_image = preprocess_image(image)
        prediction = svm_model.predict(processed_image)
        predicted_class = classes[prediction[0]]
        
        st.success(f"Loại khối u được dự đoán: {predicted_class}")
