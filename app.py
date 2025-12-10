import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder
import streamlit as st

# load the model
from keras.models import load_model

model = load_model("model_new.keras")  # load model


## open the pickle files

with open('label_encoder_gender.pkl','rb') as file:
    encode_gender=pickle.load(file)
    
with open('one_hot_encoding_geography.pkl','rb') as file:
    encode_onehot=pickle.load(file)   

with open('standard_scaler.pkl','rb') as file:
    scaler=  pickle.load(file)

## Streamlit app

st.title('Customer_Churn_Modelling')

## user input

geography=st.selectbox('Geography',encode_onehot.categories_[0])
gender=st.selectbox('Gender',encode_gender.classes_)
age=st.slider('Age',18,92)
creditscore=st.number_input('CreditScore')
tenure=st.slider('Tenure',1,10)
balance=st.number_input('Balance')
num_of_products=st.slider('NumOfProducts',1,3)
has_cr_card=st.selectbox('HasCrCard',[0,1])
is_active_member=st.selectbox('IsActiveMember',[0,1])
estimated_salary=st.number_input('EstimatedSalary')

# convert this into dataframe 

input_data=pd.DataFrame({
    'CreditScore':[creditscore],
    'Gender':[gender],
    'Age':[age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[num_of_products],
    'HasCrCard'	:[has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary' :[estimated_salary],
})

## now convert geography using one hotencoding and chnage the datatype to dataframe to concat
input_data['Gender']=encode_gender.transform(input_data['Gender'])
encoded_geography=encode_onehot.transform([[geography]]).toarray()
encoded_geography=pd.DataFrame(encoded_geography,columns=encode_onehot.get_feature_names_out())
final_input=pd.concat([input_data.reset_index(drop=True),encoded_geography],axis=1)
scaled_input=scaler.transform(final_input)
prediction=model.predict(scaled_input)
prediction_proba=prediction[0][0]
st.write(f"Probability of churn is {prediction_proba:.2f}")
if prediction_proba < 0.5:
    st.write('customer is likely to churn')
else:
    st.write('customer is not likely to churn')