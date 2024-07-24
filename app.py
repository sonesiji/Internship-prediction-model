
import streamlit as st
import pandas as pd
import pickle

# Load the dataset
data_path = "datasetnew.csv"  # Ensure the path is correct
data = pd.read_csv(data_path)

# Load the models and encoders
with open("model_title.pkl", "rb") as file:
    model_title = pickle.load(file)

with open("model_description.pkl", "rb") as file:
    model_description = pickle.load(file)

with open("model_stipend.pkl", "rb") as file:
    model_stipend = pickle.load(file)

with open("encoder_title.pkl", "rb") as file:
    encoder_title = pickle.load(file)

with open("encoder_description.pkl", "rb") as file:
    encoder_description = pickle.load(file)

# Function to predict based on user input
def predict_internship(skills, location, eligibility, duration):
    input_data = pd.DataFrame({
        'Skills Required': [skills],
        'Location': [location],
        'Eligibility': [eligibility],
        'Duration (Months)': [duration]
    })
    
    # One-hot encoding if necessary
    input_data = pd.get_dummies(input_data)
    
    # Align with the training data columns
    input_data = input_data.reindex(columns=model_title.feature_names_in_, fill_value=0)
    
    title_prediction = model_title.predict(input_data)
    description_prediction = model_description.predict(input_data)
    stipend_prediction = model_stipend.predict(input_data)
    
    internship_title = encoder_title.inverse_transform([int(title_prediction[0])])[0]
    internship_description = encoder_description.inverse_transform([int(description_prediction[0])])[0]
    stipend = stipend_prediction[0]

    # Retrieve additional details from the dataset
    matching_row = data[(data['Internship Title'] == internship_title) & (data['Internship Description'] == internship_description)].iloc[0]
    company_name = matching_row['Company Name']
    application_deadline = matching_row['Application Deadline']
    
    return internship_title, internship_description, stipend, company_name, application_deadline

# Streamlit application
st.set_page_config(page_title="Internship Predictor", layout="wide")

# Custom CSS and Bootstrap
st.markdown("""
    <link href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            background-color: #f0f2f5;
            font-family: 'Arial', sans-serif;
        }
        .sidebar .sidebar-content {
            background-color: #ffffff;
            color: #3b5998;
            padding: 1rem;
            border-right: 2px solid #3b5998;
        }
        .sidebar .sidebar-content h1, .sidebar .sidebar-content h3, .sidebar .sidebar-content label {
            color: #3b5998;
        }
        .stButton>button {
            background-color: #3b5998;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 0.5em 1em;
            cursor: pointer;
            transition: background-color 0.3s ease, transform 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #365899;
            transform: scale(1.05);
        }
        .stMarkdown {
            margin: 1em 0;
        }
        .card {
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            background-color: #ffffff;
            padding: 1em;
            border-radius: 8px;
            margin-bottom: 1em;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        .card:hover {
            transform: scale(1.02);
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
        }
        .card h4 {
            color: #3b5998;
        }
        .card p {
            color: #555555;
        }
        .main-container {
            background-color: #ffffff;
            border-radius: 8px;
            padding: 2em;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        .header-container {
            text-align: center;
            color: #4267B2;
        }
        .input-container {
            margin: 2em 0;
        }
        .prediction-results {
            margin-top: 2em;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="main-container">
        <h1 class="header-container">Internship Predictor</h1>
        <h3 class="header-container">Predict Your Ideal Internship</h3>
    </div>
""", unsafe_allow_html=True)

st.sidebar.header("Input Parameters")

# Dropdowns based on unique values from the dataset
skills_options = sorted(data['Skills Required'].unique())
location_options = sorted(data['Location'].unique())
eligibility_options = sorted(data['Eligibility'].unique())

skills = st.sidebar.selectbox("Skills Required", skills_options)
location = st.sidebar.selectbox("Location", location_options)
eligibility = st.sidebar.selectbox("Eligibility", eligibility_options)
duration = st.sidebar.number_input("Duration (Months)", min_value=1, max_value=12, step=1)

if st.sidebar.button("Predict"):
    if skills and location and eligibility and duration:
        internship_title, internship_description, stipend, company_name, application_deadline = predict_internship(skills, location, eligibility, duration)
        
        st.markdown(f"""
            <div class="main-container prediction-results">
                <h4>Prediction Results</h4>
                <div class="card">
                    <h4>Internship Title: <span style="color: #4267B2;">{internship_title}</span></h4>
                    <p><strong>Internship Description:</strong> {internship_description}</p>
                    <p><strong>Stipend (INR/Month):</strong> {stipend}</p>
                    <p><strong>Company Name:</strong> {company_name}</p>
                    <p><strong>Application Deadline:</strong> {application_deadline}</p>
                </div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.error("Please fill in all the input fields")

# Display the dataset for reference (optional)
# st.subheader("Dataset")
# st.write(data)
