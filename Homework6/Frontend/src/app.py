import streamlit as st

pg = st.navigation([
    st.Page("pages/test_endpoint_page.py"), 
    st.Page("pages/simple_page.py"), 
    st.Page("pages/history_page.py"), 
    st.Page("pages/rag_page.py"),
    st.Page("pages/filtered_rag_page.py"),
])
pg.run()