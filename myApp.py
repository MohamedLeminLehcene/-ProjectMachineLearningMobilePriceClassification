
import numpy as np
import streamlit as st

import joblib

# Load our model and scalar
model = joblib.load("C:/Users/HP/Desktop/MLProject/Impl/Mobile Price Classification/model.pkl")
scalar = joblib.load("C:/Users/HP/Desktop/MLProject/Impl/Mobile Price Classification/scalar.pkl")

def mobile_price_classification(input_data):
    # Changing the input numpy array and reshaping
    input_changed = np.array(input_data).reshape(1,-1)

    # Standardize the model
    std_input = scalar.transform(input_changed)

    prediction = model.predict(std_input)

    return "Estimated mobile price classification: " + str(prediction[0])


def main():
    # CSS styling
    st.markdown("""
    <style>
        .title {
            color: #1e3888;
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 20px;
        }
        .text-input {
            margin-bottom: 10px;
        }
        .button {
            background-color: #1e3888;
            color: #ffffff;
            font-weight: bold;
            padding: 8px 16px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        .result {
            font-weight: bold;
            margin-top: 20px;
        }
    </style>
    """, unsafe_allow_html=True)

    # Background image
    st.markdown(
        """
        <style>
            body {
                background-image: url('your_background_image_url');
                background-size: cover;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Creating the title
    st.markdown('<div class="title">Project ML Mobile Price Classification App</div>', unsafe_allow_html=True)


    col1 , col2 ,col3,col4= st.columns(4)
    input1 = col1.text_input("Battery Power")
    input2 = col2.text_input("Blue")
    input3 = col3.text_input("Clock Speed")
    input4 = col4.text_input("Dual SIM")

    col5 , col6 ,col7,col8,col9,col21= st.columns(5)

    input21 = col21.text_input("Fc")

    input5 = col5.text_input("Four g")
    input6 = col6.text_input("Int memory")
    input7 = col7.text_input("M dep")
    input8 = col8.text_input("Mobile wt")
    input9 = col9.text_input("N cores")

    col10 ,col11,col12,col13,col14= st.columns(5)

    input10 = col10.text_input("pc")
    input11 = col11.text_input("Px height")
    input12 = col12.text_input("Px width")
    input13 = col13.text_input("Ram")
    input14 = col14.text_input("Sc h")
    col15 , col16 ,col17,col18,col19= st.columns(5)
    input16 = col15.text_input("Sc w")
    input17 = col16.text_input("Talk time")
    input18 = col17.text_input("Three g")
    input19 = col18.text_input("Touch screen")
    input20 = col19.text_input("Wifi")

    # Create a button
    if st.button('Check Estimated Classifier', key="classify_button"):
        mobile_classifier = mobile_price_classification([[input1,input2,input3,input4,input21,input5,input6,input7,input8,input9,input10,input11,input12,input13,input14,input16,input17,input18,input19,input20]])  # Replace with your classifier function
        st.markdown('<div class="result">{}</div>'.format(mobile_classifier), unsafe_allow_html=True)


   

if __name__ == '__main__':
    main()






