import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Đặt st.set_page_config() là lệnh đầu tiên
st.set_page_config(
    page_title="Dự đoán Phân loại Khối U",
    page_icon="🏥",
    layout="wide"
)

# Sau đó mới đến các lệnh khác
@st.cache_resource
def load_model():
    model = joblib.load('svm_model.joblib')
    classes = joblib.load('classes.joblib')
    return model, classes

model, classes = load_model()

# CSS để tùy chỉnh giao diện
st.markdown("""
    <style>
    .big-font {
        font-size:20px !important;
        font-weight: bold;
    }
    .result-font {
        font-size:24px !important;
        font-weight: bold;
        color: #FF4B4B;
    }
    </style>
""", unsafe_allow_html=True)

# Tiêu đề
st.title("🏥 Ứng dụng Dự đoán Phân Loại Khối U")
st.markdown("---")

# Tạo giao diện nhập liệu
st.markdown('<p class="big-font">Nhập các chỉ số đặc trưng:</p>', unsafe_allow_html=True)

with st.form("prediction_form"):
    # Tạo 2 cột để layout đẹp hơn
    col1, col2 = st.columns(2)
    
    with col1:
        mean_radius = st.number_input('Mean Radius', min_value=0.0, format="%.6f")
        mean_texture = st.number_input('Mean Texture', min_value=0.0, format="%.6f")
        mean_perimeter = st.number_input('Mean Perimeter', min_value=0.0, format="%.6f")
        mean_area = st.number_input('Mean Area', min_value=0.0, format="%.6f")
        mean_smoothness = st.number_input('Mean Smoothness', min_value=0.0, format="%.6f")

    with col2:
        mean_compactness = st.number_input('Mean Compactness', min_value=0.0, format="%.6f")
        mean_concavity = st.number_input('Mean Concavity', min_value=0.0, format="%.6f")
        mean_concave_points = st.number_input('Mean Concave Points', min_value=0.0, format="%.6f")
        mean_symmetry = st.number_input('Mean Symmetry', min_value=0.0, format="%.6f")
        mean_fractal_dimension = st.number_input('Mean Fractal Dimension', min_value=0.0, format="%.6f")

    # Nút dự đoán
    submitted = st.form_submit_button("Dự đoán", use_container_width=True)

# Xử lý khi nhấn nút dự đoán
if submitted:
    # Tạo mảng đặc trưng từ input
    features = np.array([[
        mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness,
        mean_compactness, mean_concavity, mean_concave_points, mean_symmetry,
        mean_fractal_dimension
    ]])

    # Thực hiện dự đoán
    with st.spinner('Đang thực hiện dự đoán...'):
        prediction = model.predict(features)
        prediction_proba = model.predict_proba(features)

    # Hiển thị kết quả
    st.markdown("---")
    st.markdown('<p class="big-font">Kết quả dự đoán:</p>', unsafe_allow_html=True)
    
    # Hiển thị loại khối u được dự đoán
    st.markdown(f'<p class="result-font">Loại khối u: {classes[prediction[0]]}</p>', 
               unsafe_allow_html=True)

    # Hiển thị xác suất cho từng loại
    st.subheader("Xác suất dự đoán cho từng loại:")
    
    # Tạo DataFrame cho xác suất
    proba_df = pd.DataFrame({
        'Loại khối u': classes,
        'Xác suất': prediction_proba[0]
    })
    
    # Hiển thị bảng xác suất
    st.dataframe(proba_df.style.format({'Xác suất': '{:.2%}'}))
    
    # Vẽ biểu đồ xác suất
    fig_col1, fig_col2 = st.columns(2)
    with fig_col1:
        st.subheader("Biểu đồ xác suất:")
        st.bar_chart(proba_df.set_index('Loại khối u'))

# Footer
st.markdown("---")
st.markdown("Developed by QDung210 🚀")
