import pandas as pd
import tensorflow as tf
import streamlit as st
from sklearn.preprocessing import LabelEncoder, StandardScaler
# loading the trained model
model = tf.keras.model.load_model('churn.h5')


@st.cache()
# defining the function which will make the prediction using the data which the user inputs
def prediction(PhoneService, OnlineSecurity, TechSupport, DeviceProtection, Contract, PaymentMethod, tenure, MonthlyCharges, TotalCharges):
    # Pre-processing user input
    data = pd.DataFrame({'PhoneService':[PhoneService],
                         'OnlineSecurity': [OnlineSecurity],
                         'TechSupport': [TechSupport],
                         'DeviceProtection': [DeviceProtection],
                         'Contract': [Contract],
                         'PaymentMethod': [PaymentMethod],
                         'tenure': [tenure],
                         'MonthlyCharges': [MonthlyCharges],
                         'TotalCharges': [TotalCharges], })
    encoder = LabelEncoder()
    categorical = ['PhoneService', 'OnlineSecurity', 'TechSupport', 'DeviceProtection', 'Contract', 'PaymentMethod']
    for i in categorical:
        data[i] = encoder.fit_transform(data[i])

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(data)
    new_data = pd.DataFrame(scaled_features, columns=data.columns)

    prediction = model.predict(new_data)

    return prediction


# this is the main function in which we define our webpage  
def main():
    # front end elements of the web page 
    html_temp = """ 
    <div style ="background-color:yellow;padding:13px"> 
    <h1 style ="color:black;text-align:center;">Streamlit Loan Prediction ML App</h1> 
    </div> 
    """

    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html=True)

    # following lines create boxes in which user can enter data required to make prediction
    PhoneService = st.selectbox('Phone Service', ("Male", "Female"))
    OnlineSecurity = st.selectbox('Online Security', ("Unmarried", "Married"))
    TechSupport = st.number_input("Tech Support")
    DeviceProtection = st.number_input("Device Protection")
    Contract = st.selectbox('Contract', ("Unclear Debts", "No Unclear Debts"))
    PaymentMethod = st.selectbox('Payment Method', ("Unclear Debts", "No Unclear Debts"))
    tenure = st.selectbox('Tenure', ("Unclear Debts", "No Unclear Debts"))
    TotalCharges = st.selectbox('TotalCharges', ("Unclear Debts", "No Unclear Debts"))

    result = ""

    # when 'Predict' is clicked, make the prediction and store it 
    if st.button("Predict"):
        result = prediction(Gender, Married, ApplicantIncome, LoanAmount, Credit_History)
        st.success('Your loan is {}'.format(result))
        print(LoanAmount)


if __name__ == '__main__':
    main()