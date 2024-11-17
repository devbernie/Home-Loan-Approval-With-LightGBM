import gradio as gr
import joblib
import pandas as pd

# Tải mô hình đã lưu
model = joblib.load('optimized_lightgbm_model.pkl')

# Định nghĩa hàm dự đoán
def predict_loan_approval(Married, Education, ApplicantIncome, CoapplicantIncome, LoanAmount, Credit_History, Property_Area):
    # Tạo DataFrame từ các đầu vào
    input_data = pd.DataFrame({
        'Married': [Married],
        'Education': [Education],
        'ApplicantIncome': [ApplicantIncome],
        'CoapplicantIncome': [CoapplicantIncome],
        'LoanAmount': [LoanAmount],
        'Credit_History': [Credit_History],
        'Property_Area': [Property_Area]
    })
    
    # Mã hóa các cột cần thiết
    input_data['Married'] = input_data['Married'].map({'Yes': 1, 'No': 0})
    input_data['Education'] = input_data['Education'].map({'Graduate': 1, 'Not Graduate': 0})
    input_data['Property_Area_Semiurban'] = input_data['Property_Area'].apply(lambda x: 1 if x == 'Semiurban' else 0)
    input_data['Property_Area_Urban'] = input_data['Property_Area'].apply(lambda x: 1 if x == 'Urban' else 0)
    
    # Tính các đặc trưng mới
    input_data['Debt_to_Income'] = input_data['LoanAmount'] / (input_data['ApplicantIncome'] + input_data['CoapplicantIncome'] + 1)
    input_data['Total_Income'] = input_data['ApplicantIncome'] + input_data['CoapplicantIncome']
    
    # Loại bỏ cột Property_Area gốc
    input_data = input_data.drop(columns=['Property_Area'])
    
    # Dự đoán
    prediction = model.predict(input_data)
    
    return 'Phê duyệt' if prediction[0] == 1 else 'Không phê duyệt'

# Tạo giao diện Gradio
iface = gr.Interface(
    fn=predict_loan_approval,
    inputs=[
        gr.Dropdown(['Yes', 'No'], label="Married"),
        gr.Dropdown(['Graduate', 'Not Graduate'], label="Education"),
        gr.Number(label="Applicant Income"),
        gr.Number(label="Coapplicant Income"),
        gr.Number(label="Loan Amount"),
        gr.Dropdown([0, 1], label="Credit History"),
        gr.Dropdown(['Urban', 'Semiurban', 'Rural'], label="Property Area")
    ],
    outputs=gr.Textbox(label="Kết quả dự đoán"),
    title="Home Loan Approval Prediction",
    description="Nhập thông tin để dự đoán việc phê duyệt khoản vay mua nhà."
)

# Chạy ứng dụng
iface.launch()