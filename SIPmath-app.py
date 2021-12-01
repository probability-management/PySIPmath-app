#Install libs that are not preinstalled
import streamlit as st
from PIL import Image
import pandas as pd
import json
import PySIP
import requests

def convert_to_JSON(input_df,
                       filename,
                       author,
                       dependence,
                       boundedness,
                       bounds,
                       term_saved):
    PySIP.Json(input_df,
               filename,
               author,
               dependence = dependence,
               boundedness = boundedness,
               bounds = bounds,
               term_saved = term_saved)
    with open(filename) as f:
        st.download_button(
                label=f"Download {filename}",
                data=f,
                file_name=filename
                )
    return True

def sent_to_pastebin(filename,file):
    payload = {"api_dev_key" : '7lc7IMiM_x5aMUFFudCiCo35t4o0Sxx6',
    "api_paste_private" : '1',
    "api_option" : 'paste',
    "api_paste_name" : filename,
    "api_paste_expire_date" : '10M',
    "api_paste_code":file,
    "api_paste_format" : 'json'}
    url = 'https://pastebin.com/api/api_post.php'
    r = requests.post(url,data=payload)
    return r

image = Image.open('PM_logo_transparent.png')

st.image(image,width=300)
st.title('SIPmath JSON Creator')

st.sidebar.header('User Input Parameters')

# st.sidebar.markdown("""
# [Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/penguins_example.csv)
# """)
# Collects user input features into dataframe

uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"],accept_multiple_files=True)
if uploaded_file != None:
    for i,file in enumerate(uploaded_file):
        try:
            input_df = pd.read_csv(file)
        except UnicodeDecodeError:
            input_df = pd.read_csv(file,encoding='cp437')
        name = file.name.replace(".csv","")
        st.subheader(f"Preview for {name}")
        # st.table(input_df[:10].set_index(input_df.columns[0]))
        st.table(input_df[:10].assign(hack='').set_index('hack'))
        st.write("If the data above appears correct, please enter your parameters in the sidebar for this file.")
        filename = st.sidebar.text_input(f'Filename {i+1}', name+'.SIPmath',key=f"{name}_{i}_filename")
        author = st.sidebar.text_input(f'Author for {filename}', 'Unknown',key=f"{name}_author")
        dependence = st.sidebar.selectbox('Dependence', ('independent','dependent'),key=f"{name}_{i}_dependence")
        boundedness = st.sidebar.selectbox('Boundedness', ("'u' - unbounded", 
                                                           "'sl' - semi-bounded lower", 
                                                           "'su' - semi-bounded upper",
                                                           "'b' - bounded on both sides"),key=f"{name}_boundedness")
                                                           
        if boundedness == "'b' - bounded on both sides":
            #convert to int and list
            boundsl = st.sidebar.text_input('Lower Bound', '0',key=f"{name}_lower")
            boundsu = st.sidebar.text_input('Upper Bound', '1',key=f"{name}_upper")
            bounds = [int(boundsl),int(boubdsu)]
        elif boundedness.find("lower") != -1:
            bounds = [int(st.sidebar.text_input('Lower Bound', '0',key=f"{name}_lower"))]
        elif boundedness.find("upper") != -1:
            bounds = [int(st.sidebar.text_input('Upper Bound', '1',key=f"{name}_upper"))]
        else:
            bounds = [0,1]
            
        boundedness = boundedness.strip().split(" - ")[0].replace("'","")
        term_saved = st.sidebar.slider('Term Saved',3,16,3,key=f"{name}_term_saved")

        if st.sidebar.button(f'Convert to {name} SIPmath Json?',key=f"{name}_term_saved"):
            st.subheader("Preview and Download the JSON file below.")
            convert_to_JSON(input_df,
                           filename,
                           author,
                           dependence,
                           boundedness,
                           bounds,
                           term_saved)
            st.text("Copy the link below to paste into SIPmath.")
            with open(filename, 'rb') as f:
                st.write(sent_to_pastebin(filename,f.read()).text.replace("https://pastebin.com/","https://pastebin.com/raw/"))
            st.text("Mouse over the text then click on the clipboard icon to copy to your clipboard.")
            with open(filename, 'rb') as f:
                st.json(json.load(f))
else:
    input_df = pd.DataFrame()
    
#st.json() #When done I can show a preview


# if st.button('Say hello'):
    # st.write('Why hello there')
# else:
    # st.write('Goodbye')