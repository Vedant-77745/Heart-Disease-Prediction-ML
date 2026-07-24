import streamlit as st
import joblib
import pandas as pd

# Load saved files
model = joblib.load("SVM_Heart.pkl")
scaler = joblib.load("scale.pkl")
columns = joblib.load("columnns.pkl")
st.write("Scaler Feature Names:", scaler.feature_names_in_)
st.write("Model Classes:", model.classes_)


st.title("❤️ Heart Disease Prediction")

# ---------------- INPUTS ---------------- #

age = st.number_input("Age", 1, 120)
restingbp = st.number_input("Resting Blood Pressure")
chol = st.number_input("Cholesterol")
maxhr = st.number_input("Max Heart Rate")
oldpeak = st.number_input("Oldpeak")

sex = st.selectbox("Sex", ["M", "F"])
chestpain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
fastingbs = st.selectbox("Fasting Blood Sugar (1 = Yes)", [0, 1])
restingecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
exerciseangina = st.selectbox("Exercise Angina", ["Y", "N"])
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# ---------------- PREDICTION ---------------- #

if st.button("Predict"):

    # Create dictionary with all columns set to 0
    input_dict = dict.fromkeys(columns, 0)

    # Numeric values
    input_dict["Age"] = age
    input_dict["RestingBP"] = restingbp
    input_dict["Cholesterol"] = chol
    input_dict["MaxHR"] = maxhr
    input_dict["Oldpeak"] = oldpeak
    input_dict["FastingBS"] = fastingbs

    # Sex encoding
    if sex == "M":
        input_dict["Sex_M"] = 1
    else:
        input_dict["Sex_F"] = 1

    # Chest Pain encoding (ALL categories included now)
    if chestpain == "ASY":
        input_dict["ChestPainType_ASY"]=1
    elif chestpain == "ATA":
        input_dict["ChestPainType_ATA"] = 1
    elif chestpain == "NAP":
        input_dict["ChestPainType_NAP"] = 1
    elif chestpain == "TA":
        input_dict["ChestPainType_TA"] = 1
# If ASY → do nothing (all remain 0)


    # Resting ECG encoding
    if restingecg == "Normal":
        input_dict["RestingECG_Normal"] = 1
    elif restingecg == "ST":
        input_dict["RestingECG_ST"] = 1
    elif restingecg == "LVH":
        input_dict["RestingECG_LVH"] = 1

    # Exercise Angina encoding
    if exerciseangina == "Y":
        input_dict["ExerciseAngina_Y"] = 1
    else:
        input_dict["ExerciseAngina_N"] = 1

    # ST Slope encoding
    if st_slope == "Up":
        input_dict["ST_Slope_Up"] = 1
    elif st_slope == "Flat":
        input_dict["ST_Slope_Flat"] = 1
    elif st_slope == "Down":
        input_dict["ST_Slope_Down"] = 1

    # Convert to DataFrame in correct order
    input_df = pd.DataFrame([input_dict])

# 🚨 VERY IMPORTANT
    input_df = input_df.reindex(columns=columns)



    # Scale input
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)

    # Output result
    if prediction[0] == 1:
        st.error("⚠ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")
