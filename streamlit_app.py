import streamlit as st 
st.markdown(
    "<h1 style='color:#C5B358; font-weight:700;'>Tony DiRubbo's Homework Manager for 688</h1>",
    unsafe_allow_html=True
)
st.text('Select a Homework in the Sidebar. Homework is a collection of assignments from IST: 688 - Building Human Centered AI Applications') 
hw1_page = st.Page("HWs/HW1.py", title = "Homework 1") 
hw2_page = st.Page("HWs/HW2.py", title = "Homework 2") 
pg = st.navigation([hw2_page,hw1_page]) 
pg.run()