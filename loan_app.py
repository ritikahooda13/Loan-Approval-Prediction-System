import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Page Configuration
st.set_page_config(page_title="Loan Approval Prediction System", layout="wide")

# UI Title Header
st.title("🏦 Loan Approval Prediction System")
st.write("An end-to-end predictive framework to assess applicant risk metrics and determine loan eligibility status.")

# --- 1. Robust Dataset Generation & Preprocessing ---
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
    
    # Internal deterministic logic mapping to guide machine learning behavior
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

# --- Sidebar Controls ---
st.sidebar.header("Navigation Control Panel")
menu = st.sidebar.selectbox("Jump to Section", ["Dataset Summary", "Exploratory Plots", "Model Insights", "Interactive Prediction Engine"])

# --- SECTION 1: Dataset Summary ---
if menu == "Dataset Summary":
    st.header("📋 Baseline Data Audit & Metrics")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records Ingested", df.shape[0])
    col2.metric("Total Observed Dimensions", df.shape[1])
    col3.metric("Missing Attributes Managed", df.isnull().sum().sum())
    
    st.write("### Data Samples (First 10 Records)")
    st.dataframe(df.head(10), use_container_width=True)
    
    st.write("### Continuous Metrics Summary")
    st.dataframe(df.describe().fillna('-'), use_container_width=True)

# --- SECTION 2: Exploratory Plots ---
elif menu == "Exploratory Plots":
    st.header("📊 Automated Data Visualization Profiles")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### Continuous Features: Annual Income Spread")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df['Annual_Income'], kde=True, color='#2E7D32', ax=ax)
        st.pyplot(fig)
        
    with col2:
        st.write("### Target Distribution: Application Status Split")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(x='Loan_Status', data=df, palette='Greens_r', ax=ax)
        st.pyplot(fig)

    st.write("### Categorical Interactions: Credit Standing vs Target Output")
    fig, ax = plt.subplots(figsize=(8, 3.5))
    sns.countplot(x='Credit_History', hue='Loan_Status', data=df, palette='YlGnBu', ax=ax)
    st.pyplot(fig)

# --- SECTION 3: Model Insights ---
elif menu == "Model Insights":
    st.header("⚙️ Predictive Diagnostics & Evaluation")
    
    X = df_encoded.drop(columns=['Applicant_ID', 'Loan_Status'])
    y = df_encoded['Loan_Status'].map({'Approved': 1, 'Rejected': 0})
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    st.metric("Global Accuracy Baseline", f"{accuracy * 100:.2f}%")
    
    st.write("### Performance Report Breakdown")
    st.json(classification_report(y_test, y_pred, output_dict=True))
    
    st.write("### Relative Feature Importances")
    feature_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=feature_importance.values, y=feature_importance.index, palette='viridis', ax=ax)
    st.pyplot(fig)

# --- SECTION 4: Interactive Prediction Engine ---
elif menu == "Interactive Prediction Engine":
    st.header("🔮 System Evaluation Inference Portal")
    st.write("Modify real-time input fields below to test operational classification outcomes.")
    
    with st.form("inference_input_form"):
        c1, c2 = st.columns(2)
        with c1:
            gender = st.selectbox("Applicant Gender Type", ["Male", "Female"])
            married = st.selectbox("Marital History Status", ["Yes", "No"])
            education = st.selectbox("Academic Attainment Level", ["Graduate", "Not Graduate"])
            employment = st.selectbox("Current Professional Status", ["Employed", "Self-Employed"])
        with c2:
            income = st.number_input("Calculated Annual Income ($)", min_value=10000, max_value=500000, value=65000)
            loan_amt = st.number_input("Requested Principal Loan Value ($)", min_value=5000, max_value=300000, value=35000)
            credit = st.selectbox("Valid Historical Credit Score Status", [1.0, 0.0])
            property_area = st.selectbox("Zoning Demographics Designation", ["Urban", "Semi-Urban", "Rural"])
            
        submit_btn = st.form_submit_button("Query Machine Learning Model")
        
    if submit_btn:
        input_payload = pd.DataFrame([{
            'Gender': label_encoders['Gender'].transform([gender])[0],
            'Married': label_encoders['Married'].transform([married])[0],
            'Education': label_encoders['Education'].transform([education])[0],
            'Employment_Status': label_encoders['Employment_Status'].transform([employment])[0],
            'Annual_Income': income,
            'Loan_Amount': loan_amt,
            'Credit_History': credit,
            'Property_Area': label_encoders['Property_Area'].transform([property_area])[0]
        }])
        
        X_all = df_encoded.drop(columns=['Applicant_ID', 'Loan_Status'])
        y_all = df_encoded['Loan_Status'].map({'Approved': 1, 'Rejected': 0})
        
        final_model = RandomForestClassifier(random_state=42)
        final_model.fit(X_all, y_all)
        
        outcome = final_model.predict(input_payload)[0]
        st.write("---")
        if outcome == 1:
            st.success("🎉 **Application Evaluation Result: APPROVED** — The candidate parameters fall securely within automated credit risk boundaries.")
        else:
            st.error("❌ **Application Evaluation Result: REJECTED** — Parameters display elevated non-compliance markers relative to data history distributions.")