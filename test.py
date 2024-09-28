# Importing Libraries
import pickle
import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import numpy as np
from PIL import Image

# Setting Webpage Configurations
st.title("***:red[Industrial Copper Modeling]***")
st.subheader("***:green[The project demonstrates sales and successful ratio in copper industry]***")
import base64
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()
background_image_path = r'C:\Users\Hp\OneDrive\Desktop\Industrialcopper\copper4.jpg'
base64_image = get_base64_of_bin_file(background_image_path)
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-image: url("data:image/jpg;base64,{base64_image}");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)
statement=st.radio(":red[select the option:]",[":grey[Project info]",":grey[Data Visualisation]",":grey[Copper estimation]",":grey[Project conclusion]"])

def load_model():
    model = pickle.load(open('price_prediction.pkl','rb'))
    return model
def load_encoder():
    encoder = pickle.load(open('encoder.pkl','rb'))
    return encoder

def load_class_model():
    model = pickle.load(open('Status_prediction.pkl','rb'))
    return model


# Price prediction model
reg_model = load_model()

# encoder
encoder = load_encoder()

# Status Prediction model
class_model = load_class_model()

# Reading the Dataframe
model_df = pd.read_csv(r'Copper.csv')

# Querying Win/Lost status
query_df = model_df.query("status == 'Won' or status == 'Lost'")

if statement==':grey[Project info]':
    st.subheader("**Introduction**")
    st.write("""Like many other industries, the copper sector struggles to deal with less complicated but distorted and noisy sales and pricing data. 
            Manual forecasting takes time and might not be precise. Making use of machine learning techniques can greatly enhance decision-making. 
            We will deal with problems like skewness and noisy data in this solution, and create a regression model to forecast selling
            rates and a classification algorithm to forecast the lead status (WON or LOST).""")
    st.subheader("Domain: :grey[Manufacturing]")
    st.subheader("**Key Objectives:**")
    st.markdown("""
        1. **Data Exploration:**
            - Identify and address skewness and outliers in the sales dataset.
        2. **Data Preprocessing:**
            - Transform data and implement strategies to handle missing values effectively.
        3. **Regression Model:**
            - Develop a robust regression model to predict '**Selling_Price.**'
            - Utilize advanced techniques such as data normalization and feature scaling.
        4. **Classification Model:**
            - Build a classification model to predict lead status (WON/LOST).
            - Leverage the '**STATUS**' variable, considering WON as Success and LOST as Failure.""")
    st.subheader("**Tools Used:**")
    st.markdown("""
        - **_Python:_** Facilitates versatile programming capabilities.
        - **_Pandas and NumPy:_** These libraries will be used for data manipulation and preprocessing.
        - **_Scikit-Learn:_** A powerful machine learning library that includes tools for regression and classification models.
        - **_Streamlit:_** A user-friendly library for creating web applications with minimal code, perfect for building an interactive interface for our models.""")
if statement==":grey[Data Visualisation]":
    col1,col2=st.columns(2)
    with col1:
        fig = px.bar(model_df['country'].value_counts(),title="Sales done in  num of countries ")
        st.plotly_chart(fig)
    with col2:
        fig1 = px.bar(model_df['status'].value_counts(),title="Status of sales")
        st.plotly_chart(fig1)
        
    col3,col4=st.columns(2)
    with col3:
        fig3 = px.bar(model_df['item type'].value_counts(),title="Item of sales")
        st.plotly_chart(fig3)
    with col4:
        corr_df = model_df[['item_date','thickness','width','delivery date','selling_price','quantity_tons']].corr(numeric_only=True)
        fig4= px.imshow(corr_df,text_auto = True)
        st.plotly_chart(fig4)
        
if statement==':grey[Copper estimation]':
    tab1,tab2 = st.tabs(['Selling Price Prediction','Status Prediction'])

    with tab1:

        item_year = st.selectbox('Select the Item year',options = query_df['item_date'].value_counts().index.sort_values())

        country = st.selectbox('Select the Country Code',options = query_df['country'].value_counts().index.sort_values())

        item_type = st.selectbox('Select the Item type',options = query_df['item type'].unique())

        application = st.selectbox('Select the Application number',options = query_df['application'].value_counts().index.sort_values())

        product_ref = st.selectbox('Select the Product Category',options = query_df['product_ref'].value_counts().index.sort_values())

        delivery_year = st.selectbox('Select the Delivery year',options = query_df['delivery date'].value_counts().index.sort_values())

        thickness = st.number_input('Enter the Thickness')
        log_thickness = np.log(thickness)

        width = st.number_input('Enter the width')
        log_width = np.log(width)

        quantity_tons = st.number_input('Enter the Quantity (in tons)')
        log_quantity = np.log(quantity_tons)

        submit = st.button('Predict Price')

        if submit:
        
            user_input = pd.DataFrame([[item_year,country,item_type,application,log_thickness,log_width,product_ref,delivery_year,quantity_tons]],
                                columns = ['item_date','country','item type','application','thickness','width','product_ref','delivery date','quantity_tons'])
            
            prediction = reg_model.predict(user_input)
            
            selling_price = np.exp(prediction)
            st.subheader(f':green[Predicted Price] : {round(selling_price[0])}')

    with tab2:

        country = st.selectbox('Select any one Country Code',options = query_df['country'].value_counts().index.sort_values())

        item_type = st.selectbox('Select any one Item type',options = query_df['item type'].unique())

        product_ref = st.selectbox('Select any one Product Category',options = query_df['product_ref'].value_counts().index.sort_values())

        delivery_year = st.selectbox('Select a Delivery year',options = query_df['delivery date'].value_counts().index.sort_values())

        thickness = st.number_input('Enter an Thickness')
        log_thickness = np.log(thickness)

        width = st.number_input('Enter an width')
        log_width = np.log(width)

        selling_price = st.number_input('Enter an Selling Price')
        log_selling_price = np.log(selling_price)

        quantity_tons = st.number_input('Enter an Quantity (in tons)')
        log_quantity = np.log(quantity_tons)

        user_input_1 = pd.DataFrame([[country,item_type,log_thickness,log_width,product_ref,delivery_year,log_selling_price,log_quantity]],
                        columns = ['country','item type','thickness','width','product_ref','delivery date','selling_price','quantity_tons'])
        
        submit1 = st.button('Predict')

        if submit1:
            transformed_data = encoder.transform(user_input_1)
            prediction = class_model.predict(transformed_data)
            
            if prediction[0] == 1:
                st.subheader(':green[Predicted Status] : Won')
            else:
                st.subheader(':green[Predicted Status] : Lost')
                
if statement==":grey[Project conclusion]":
    st.subheader("**_Key Insights - What I learn from the Project_**")
    st.write("Manually addressing these issues is time-consuming and may not yield optimal pricing decisions.")
    st.subheader("**_OverView_**")
    st.markdown("""By incorporating machine learning in data exploration, preprocessing, regression, and classification, this solution provides a comprehensive approach for the copper industry to improve pricing decisions and lead status assessments. 
                The Streamlit web application is a useful tool that guarantees decision-makers' accessibility and usability, with a focus on the special tasks of **_Selling Price_** and **_Stauts Lead_** prediction.""")
    button = st.button("EXIT!")
    if button:
        st.success("**Thank you for utilizing this platform. I hope you have received the predicted price and status for your copper industry!❤️**")
