import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

st.set_page_config(page_title="Loan Approval Prediction System", layout="wide")
st.title("🏦 Loan Approval Prediction System")
st.write("Analyze applicant attributes and predict loan approval status.")

@st.cache_data
def load_and_preprocess_data():
    np.random.seed(42)
    n_samples = 300
    
    data = {
        'Applicant_ID': [f"LP{i:03d}" for i in range(1, n_samples + 1)],
        'Gender': np.random.choice(['Male', 'Female'], n_samples, p=[0.7, 0.3]),
        'Married': np.random.choice(['Yes', 'No'], n_samples, p=[0.6, 0.4]),
        'Education': np.random.choice(['Graduate', 'Not Graduate'], n_samples, p=[0.8, 0.2]),
        'Employment_Status': np.random.choice(['Self-Employed', 'Employed'], n_samples, p=[0.15, 0.85]),
        'Annual_Income': np.random.randint(30000, 150000, n_samples),
        'Loan_Amount': np.random.randint(10000, 80000, n_samples),
        'Credit_History': np.random.choice([1.0, 0.0], n_samples, p=[0.8, 0.2]),
        'Property_Area': np.random.choice(['Urban', 'Semi-Urban', 'Rural'], n_samples)
    }
    
    df = pd.DataFrame(data)
    df['Loan_Status'] = np.where(
        (df['Credit_History'] == 1.0) & (df['Annual_Income'] * 0.5 > df['Loan_Amount'] * 0.1),
        'Approved', 'Rejected'
    )
    
    df_encoded = df.copy()
    label_encoders = {}
    categorical_cols = ['Gender', 'Married', 'Education', 'Employment_Status', 'Property_Area']
    
    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        label_encoders[col] = le
        
    return df, df_encoded, label_encoders

df, df_encoded, label_encoders = load_and_preprocess_data()

menu = st.sidebar.selectbox("Navigate Project Sections", ["Dataset Overview", "Exploratory Data Analysis", "Model Training & Evaluation", "Loan Prediction Interface"])

if menu == "Dataset Overview":
    st.header("📋 Dataset Management & Preprocessing")
    st.write("### Raw Dataset Sample")
    st.dataframe(df.head(10))
    
    st.write("### Dataset Information")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Rows", df.shape[0])
    col2.metric("Total Columns", df.shape[1])
    col3.metric("Missing Values", df.isnull().sum().sum())

elif menu == "Exploratory Data Analysis":
    st.header("📊 Exploratory Data Analysis (EDA)")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### Income Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df['Annual_Income'], kde=True, color='skyblue', ax=ax)
        st.pyplot(fig)
        
    with col2:
        st.write("### Approval Rate Comparison")
        fig, ax = plt.subplots()
        sns.countplot(x='Loan_Status', data=df, palette='Set2', ax=ax)
        st.pyplot(fig)

elif menu == "Model Training & Evaluation":
    st.header("⚙️ Model Training & Performance Metrics")
    X = df_encoded.drop(columns=['Applicant_ID', 'Loan_Status'])
    y = df_encoded['Loan_Status'].map({'Approved': 1, 'Rejected': 0})
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    st.metric("Model Prediction Accuracy", f"{acc * 100:.2f}%")
    
    st.write("### Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.json(report)

elif menu == "Loan Prediction Interface":
    st.header("🔮 Real-Time Loan Eligibility Checker")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            married = st.selectbox("Marital Status", ["Yes", "No"])
            education = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
            employment = st.selectbox("Employment Status", ["Employed", "Self-Employed"])
        with col2:
            income = st.number_input("Annual Income ($)", min_value=10000, max_value=500000, value=50000)
            loan_amt = st.number_input("Requested Loan Amount ($)", min_value=5000, max_value=300000, value=25000)
            credit = st.selectbox("Credit History Score", [1.0, 0.0])
            property_area = st.selectbox("Property Area Type", ["Urban", "Semi-Urban", "Rural"])
            
        submit = st.form_submit_button("Predict Application Status")
        
    if submit:
        input_data = pd.DataFrame([{
            'Gender': label_encoders['Gender'].transform([gender])[0],
            'Married': label_encoders['Married'].transform([married])[0],
            'Education': label_encoders['Education'].transform([education])[0],
            'Employment_Status': label_encoders['Employment_Status'].transform([employment])[0],
            'Annual_Income': income,
            'Loan_Amount': loan_amt,
            'Credit_History': credit,
            'Property_Area': label_encoders['Property_Area'].transform([property_area])[0]
        }])
        
        X = df_encoded.drop(columns=['Applicant_ID', 'Loan_Status'])
        y = df_encoded['Loan_Status'].map({'Approved': 1, 'Rejected': 0})
        model = RandomForestClassifier(random_state=42)
        model.fit(X, y)
        
        prediction = model.predict(input_data)[0]
        st.write("---")
        if prediction == 1:
            st.success("🎉 **Application Approved!**")
        else:
            st.error("❌ **Application Rejected.**")