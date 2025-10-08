import streamlit as st 
st.markdown(
    "<h1 style='color:#C5B358; font-weight:700;'>Tony DiRubbo's Homework Manager for 688</h1>",
    unsafe_allow_html=True
)
st.text('Select a Homework in the Sidebar. Homework is a collection of assignments from IST: 688 - Building Human Centered AI Applications') 
hw1_page = st.Page("HWs/HW1.py", title = "Homework 1") 
hw2_page = st.Page("HWs/HW2.py", title = "Homework 2") 
hw3_page = st.Page("HWs/HW3.py", title = "Homework 3") 
hw4_page = st.Page("HWs/HW4.py", title = "Homework 4") 
hw5_page = st.Page("HWs/HW5.py", title = "Homework 5") 
hw7_page = st.Page("HWs/HW7.py", title = "Homework 7") 



pg = st.navigation([hw7_page,hw5_page,hw4_page,hw3_page,hw2_page,hw1_page]) 
pg.run()