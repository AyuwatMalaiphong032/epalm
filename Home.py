import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import pickle

st.image('./pic/312622237.png')

html_8="""
<div style="background-color:#66CCCC;padding:15px;border-radius:15px 15px 15px 15px;border-style:'solid';border-color:Bisque">
<center><h5>การพยากรณ์การบริจาคโดยใช้ Decision tree</h5></center>
</div>
"""

st.markdown(html_8,unsafe_allow_html=True)
st.markdown("")

dt=pd.read_csv("./data/transfusion.csv")
st.write(dt.head(10))

data1 = dt['Recency '].sum()
data2 = dt['Frequency '].sum()
data3 = dt['Monetary '].sum()
data4 = dt['Time '].sum()
data5 = dt['donated '].sum()

dx=[data1,data2,data3,data4,data5]
dx2=pd.DataFrame(dx, index=["d1", "d2", "d3", "d4", "d5"])

if st.button("แสดงการจินตทัศน์ข้อมูล"):
   st.area_chart(dx2)
   st.button("ไม่แสดงข้อมูล")
else:
    st.write("ไม่แสดงข้อมูล")

html_8="""
<div style="background-color:#66CCCC;padding:15px;border-radius:15px 15px 15px 15px;border-style:'solid';border-color:Bisque">
<center><h5>การทำนายข้อมูล</h5></center>
</div>
"""
st.markdown(html_8,unsafe_allow_html=True)
st.markdown("")

Recency=st.number_input("กรุณาเลือกข้อมูล Recency")
Frequency=st.number_input("กรุณาเลือกข้อมูล Frequency")
Monetary= st.number_input("กรุณาเลือกข้อมูล Monetary")
Time= st.number_input("กรุณาเลือกข้อมูล Time")
donated= st.number_input("กรุณาเลือกข้อมูล donated")

if st.button("ทำนายผล"):
   loaded_model = pickle.load(open('./data/blood_model.sav', 'rb'))
   input_data =  (Recency,Frequency,Monetary,Time,donated)
   # changing the input_data to numpy array
   input_data_as_numpy_array = np.asarray(input_data)
   # reshape the array as we are predicting for one instance
   input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
   prediction = loaded_model.predict(input_data_reshaped)
   st.write(prediction)
   if prediction == 'Sad':
        st.image('./pic/746792-200.png')
   elif prediction == 'Ok':
        st.image('./pic/12.jpg')
   else:
        st.image('./pic/Thankyou.png')
   st.button("ไม่แสดงข้อมูล")
else:
    st.write("ไม่แสดงข้อมูล")
