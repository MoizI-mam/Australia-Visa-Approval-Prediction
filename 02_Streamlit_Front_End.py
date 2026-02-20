import joblib 
import streamlit as st
import pandas as pd

# First I unpickle all my picckle file
model = joblib.load("logistic_regression.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

# First I set my Web Page tile icon etc
st.set_page_config(page_title="Australia Visa Predictor", page_icon="🛂", layout="centered")


# Now I set my website Heading
st.title("🛂 Australia Visa Approval Predictor")
st.markdown("### Built by **Moiz Imam**")
st.markdown("Fill in the details below to predict your visa approval probability.")

# Now I Take Inputs from User
age = st.slider("Age",1,100,25)

# Now Document Input 
st.subheader("📄 Tell Us What Documents You Have!")

bank_statement = 1 if st.checkbox("Bank Statement") else 0
birth_certificate = 1 if st.checkbox("Birth Certificate") else 0
english_proficiency = 1 if st.checkbox("English Proficiency") else 0
invitation_letter = 1 if st.checkbox("Invitation Letter")   else 0
itinerary = 1 if st.checkbox("Itinerary") else 0
job_offer_letter = 1 if st.checkbox("Job Offer Letter") else 0
medical_certificate = 1 if st.checkbox("Medical Certificate") else 0
offer_letter = 1 if st.checkbox("Offer Letter") else 0
passport = 1 if st.checkbox("Passport") else 0
police_clearance = 1 if st.checkbox("Police Clearance") else 0
travel_insurance = 1 if st.checkbox("Travel Insurance") else 0
work_experience = 1 if st.checkbox("Work Experience Letters") else 0

# Now last 2 inputs
gender = st.selectbox("Gender",["Female","Male","Other"])
visa_type = st.selectbox("Visa Type",["Student Visa","Visit Visa","Work Visa"])

if st.button("Predict"):
    
    input_data = {
        'age'                     : age,
        'Bank_Statement'          : bank_statement,
        'Birth_Certificate'       : birth_certificate,
        'English_Proficiency'     : english_proficiency,
        'Invitation_Letter'       : invitation_letter,
        'Itinerary'               : itinerary,
        'Job_Offer_Letter'        : job_offer_letter,
        'Medical_Certificate'     : medical_certificate,
        'Offer_Letter'            : offer_letter,
        'Passport'                : passport,
        'Police_Clearance'        : police_clearance,
        'Travel_Insurance'        : travel_insurance,
        'Work_Experience_Letters' : work_experience,
        'gender_Male'             : 1 if gender == "Male" else 0,
        'gender_Other'            : 1 if gender == "Other" else 0,
        'visa_type_Visit Visa'    : 1 if visa_type == "Visit Visa" else 0,
        'visa_type_Work Visa'     : 1 if visa_type == "Work Visa" else 0,
    }
    input_df = pd.DataFrame([input_data])
    input_df['age'] = scaler.transform(input_df[['age']])
    prob = model.predict_proba(input_df)[0][1] * 100

    st.markdown(f"### You Have **{prob:.1f}%** Chance of Visa Approval!")

    if prob >= 50:
        st.success("✅ Visa Likely Approved!")
    else:
        st.error("❌ Visa Likely Rejected!")