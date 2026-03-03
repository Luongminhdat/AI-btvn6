import streamlit as st
import pandas as pd
import pickle

# 1. Load mô hình đã lưu
@st.cache_resource
def load_model():
    with open('xgboost_pipeline.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()

st.title("Dự đoán Nguy cơ Cảnh báo học vụ 🎓")
st.write("Nhập thông tin sinh viên để hệ thống đánh giá.")

# 2. Tạo form nhập liệu cho giao diện
with st.form("student_form"):
    # --- CÁC CỘT TEXT ---
    advisor_notes = st.text_area("Advisor Notes (Ghi chú của cố vấn)", "")
    personal_essay = st.text_area("Personal Essay (Bài luận)", "")
    
    # --- CÁC CỘT NUMERIC (SỐ) ---
    # Ví dụ: Sửa 'GPA', 'Absences' thành tên cột số thực tế trong file train.csv
    gpa = st.number_input("Điểm GPA", min_value=0.0, max_value=4.0, value=3.0)
    # absences = st.number_input("Số buổi vắng", value=0)
    
    # --- CÁC CỘT CATEGORICAL (PHÂN LOẠI) ---
    # Ví dụ: Sửa 'Gender' thành tên cột phân loại thực tế
    gender = st.selectbox("Giới tính", ["Male", "Female"])
    
    submitted = st.form_submit_button("Dự đoán trạng thái")

if submitted:
    # 3. Đóng gói dữ liệu đầu vào thành DataFrame
    # TÊN CỘT PHẢI VIẾT HOA/THƯỜNG Y HỆT FILE TRAIN.CSV
    input_data = pd.DataFrame({
        'Advisor_Notes': [advisor_notes],
        'Personal_Essay': [personal_essay],
        'GPA': [gpa],          # Sửa lại cho đúng tên cột
        'Gender': [gender],    # Sửa lại cho đúng tên cột
        # ... THÊM ĐẦY ĐỦ CÁC CỘT CÒN LẠI VÀO ĐÂY ...
    })
    
    try:
        # 4. Dự đoán qua Pipeline (nó sẽ tự tự động làm TF-IDF, fillna, OneHot cho input này)
        prediction = model.predict(input_data)[0]
        
        # 5. Hiển thị kết quả
        status_map = {
            0: "✅ Normal (Sinh viên học tập bình thường)", 
            1: "⚠️ Academic Warning (Cảnh báo học vụ)", 
            2: "🚨 Dropout (Nguy cơ thôi học rất cao)"
        }
        
        st.subheader("Kết quả dự đoán:")
        if prediction == 0:
            st.success(status_map[prediction])
        elif prediction == 1:
            st.warning(status_map[prediction])
        else:
            st.error(status_map[prediction])
            
    except Exception as e:
        st.error(f"Lỗi: {e}. Bạn hãy kiểm tra lại xem trong phần pd.DataFrame đã truyền đủ tên cột giống file train.csv chưa nhé!")