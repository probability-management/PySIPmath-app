#Install libs that are not preinstalled
import streamlit as st
from PIL import Image
import pandas as pd
import json
import PySIP
import requests
import numpy as np
from metalog import metalog
import matplotlib.pyplot as plt

#Taken from the metalog
def plot(m, norm=True):
    
    # Collecting data to set limits of axes
    res_data = pd.DataFrame({'term': np.repeat(str(m['params']['term_lower_bound']) \
                                                     + ' Terms', len(m['M'].iloc[:, 0])),
                                   'pdfValues': m['M'].iloc[:, 0],
                                   'quantileValues': m['M'].iloc[:, 1],
                                   'cumValue': m['M']['y']
                                   })
    if m['M'].shape[-1] > 3:
        for i in range(2, len(m['M'].iloc[0, ] - 1) // 2 + 1):
            temp_data = pd.DataFrame({'term': np.repeat(str(m['params']['term_lower_bound'] + i - 1) \
                                                          + ' Terms', len(m['M'].iloc[:, 0])),
                                        'pdfValues': m['M'].iloc[:, i * 2 - 2],
                                        'quantileValues': m['M'].iloc[:, i * 2 - 1],
                                        'cumValue': m['M']['y']})
            res_data = pd.concat([res_data, temp_data], ignore_index=True)
    
    # Collecting data into dictionary
    InitialResults = {}
    InitialResults[str(m['params']['term_lower_bound']) + ' Terms'] = pd.DataFrame({
            'pdfValues': m['M'].iloc[:, 0],
            'quantileValues': m['M'].iloc[:, 1],
            'cumValue': m['M']['y']
            })
    
    if m['M'].shape[-1] > 3:
        for i in range(2, len(m['M'].iloc[0, ] - 1) // 2 + 1):
            InitialResults[str(m['params']['term_lower_bound'] + i - 1) + ' Terms'] = pd.DataFrame({
                    'pdfValues': m['M'].iloc[:, i * 2 - 2],
                    'quantileValues': m['M'].iloc[:, i * 2 - 1],
                    'cumValue': m['M']['y']
                    })
    
    # ggplot style
    plt.style.use('ggplot')
    
    results_len = len(InitialResults)
    # fig, ax = plt.subplots(results_len, 2, figsize=(8, 3*results_len), sharex='col')
    fig, ax = plt.subplots(5, 3, figsize=(8, 3*3), sharex='col')
    # i = 2
    for i in range(2, 6 + 1):
        for j in range(0,3):
            current_term = (2 + (i - 2)*3 + j) 
            if results_len + 2 > current_term and m['Validation']['valid'][current_term] == 'yes':# Check to make sure it is valid before plotting.
                # Plotting PDF
                ax[i-2, j].plot(InitialResults[str(current_term) + ' Terms']['quantileValues'], InitialResults[str(current_term) + ' Terms']['pdfValues'],
                      linewidth=2,c='darkblue')

                # Plotting CDF
                # ax[i-2, j].plot(InitialResults[str(current_term) + ' Terms']['quantileValues'], InitialResults[str(current_term) + ' Terms']['cumValue'],
                      # linewidth=2)
                # Plot data 
                # ax[i-2, j].scatter(m["dataValues"]['x'],m["dataValues"]['probs'],c='black',edgecolor='white')
            else: #if not valid plot nothing
                #Plotting blank PDF chart
                # ax[i-2, 0].plot()
                # Plotting blank CDF chart
                ax[i-2, j].plot()
            #Axes setup    
            # if norm:
            # ax[i-2, j].axis([min(res_data['quantileValues']), max(res_data['quantileValues']),
                  # round(min(m["dataValues"]['probs']),1), round(max(m["dataValues"]['probs']),1)]) 
            ax[i-2, j].patch.set_facecolor('white')
            ax[i-2, j].axes.xaxis.set_ticks([])     
            ax[i-2, j].axes.yaxis.set_ticks([])  
            if current_term != 5*3:
                ax[i-2, j].set(title=str(current_term) + ' Terms', ylabel='PDF')
                # ax[i-2, j].patch.set()
            else:
               ax[i-2, j].set(title=str(current_term) + ' Terms', ylabel='PDF', xlabel='Quantiles')
               
               # ax[i-2, j].patch.set(title=str(current_term) + ' Terms', ylabel='PDF', xlabel='Quantiles')
                  
                      
            # if current_term != 5*3:
                # ax[i-2, j].set(title=str(current_term) + ' Terms', ylabel='CDF')
            # else:
               # ax[i-2, j].set(title=str(current_term) + ' Terms', ylabel='CDF', xlabel='Quantiles')
                  
    plt.tight_layout()
    # plt.show()
    st.pyplot(plt)

def convert_to_JSON(input_df,
                       filename,
                       author,
                       dependence,
                       boundedness,
                       bounds,
                       term_saved,
                       probs):

    PySIP.Json(input_df,
               filename,
               author,
               dependence = dependence,
               boundedness = boundedness,
               bounds = bounds,
               term_saved = term_saved,
               probs=probs
               )
    
    with open(filename) as f:
        st.download_button(
                label=f"Download {filename}",
                data=f,
                file_name=filename
                )
    return True

def preprocess_charts(x,
                       probs,
                       boundedness,
                       bounds):
	#Create metalog
	# st.write(boundedness,
                       # bounds)
	mfitted = metalog.fit(x, bounds = bounds, boundedness = boundedness, fit_method='OLS', term_limit = len(x), probs=probs)
    #Create graphs
	# plot(mfitted)
	st.write(mfitted.keys())
	st.write(mfitted)
	st.write(mfitted['M'])
	st.write(mfitted["dataValues"])
	st.write(mfitted['Validation'])

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

def input_data(name,i,df,probs=None):
    if probs is None:
        probs = np.nan
        max_val = 16
        default_val = 5
    else:
        max_val = df.shape[0]
        default_val = max_val
    st.write("If the data above appears correct, please enter your parameters in the sidebar for this file.")
    filename = st.sidebar.text_input(f'Filename {i+1}', name+'.SIPmath',key=f"{name}_{i}_filename")
    author = st.sidebar.text_input(f'Author for {filename}', 'Unknown',key=f"{name}_author")
    dependence = st.sidebar.selectbox('Dependence', ('independent','dependent'),key=f"{name}_{i}_dependence")
    boundedness = st.sidebar.selectbox('Boundedness', ("'u' - unbounded", 
                                                       "'sl' - semi-bounded lower", 
                                                       "'su' - semi-bounded upper",
                                                       "'b' - bounded on both sides"),key=f"{name}_boundedness")
                                                       
    if boundedness == "'b' - bounded on both sides":
        #convert to float and list
        boundsl = st.sidebar.text_input('Lower Bound', '0',key=f"{name}_lower")
        boundsu = st.sidebar.text_input('Upper Bound', '1',key=f"{name}_upper")
        bounds = [float(boundsl),float(boundsu)]
    elif boundedness.find("lower") != -1:
        bounds = [float(st.sidebar.text_input('Lower Bound', '0',key=f"{name}_lower"))]
    elif boundedness.find("upper") != -1:
        bounds = [float(st.sidebar.text_input('Upper Bound', '1',key=f"{name}_upper"))]
    else:
        bounds = [0,1]
        
    boundedness = boundedness.strip().split(" - ")[0].replace("'","")
    if max_val > 3:
        term_saved = st.sidebar.slider('Term Saved',3,max_val,default_val,key=f"{name}_term_saved")

    if st.sidebar.button(f'Convert to {name} SIPmath Json?',key=f"{name}_term_saved"):
        st.subheader("Preview and Download the JSON file below.")
        convert_to_JSON(df,
                       filename,
                       author,
                       dependence,
                       boundedness,
                       bounds,
                       term_saved,
                       probs=probs)
        # st.text("Copy the link below to paste into SIPmath.")
        # with open(filename, 'rb') as f:
            # st.write(sent_to_pastebin(filename,f.read()).text.replace("https://pastebin.com/","https://pastebin.com/raw/"))
        st.text("Mouse over the text then click on the clipboard icon to copy to your clipboard.")
        with open(filename, 'rb') as f:
            st.json(json.load(f))
    
image = Image.open('PM_logo_transparent.png')

st.image(image,width=1000)
#st.title('SIPmath JSON Creator')
st.sidebar.header('User Input Parameters')

# Collects user input features into dataframe
data_type = st.sidebar.radio('Input Data Type', ('CSV File','Quantile'), index=0)

if data_type == 'CSV File':
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"],accept_multiple_files=True)
    if uploaded_file != None:
        for i,file in enumerate(uploaded_file):
            try:
                input_df = pd.read_csv(file)
            except UnicodeDecodeError:
                input_df = pd.read_csv(file,encoding='cp437')
            name = file.name.replace(".csv","")
            st.subheader(f"Preview for {name}")
            st.write(input_df[:10].to_html(index=False), unsafe_allow_html=True)
            input_data(name,i,input_df)
    else:
        input_df = pd.DataFrame()
elif data_type == 'Quantile':
    reference_probabilities = {
                                    2:[0.250,0.750],
                                    3:[0.100,0.500,0.900],
                                    4:[0.100,0.250,0.500,0.750],
                                    5:[0.100,0.250,0.500,0.750,0.900],                                    
                                    6:[0.010,0.100,0.250,0.500,0.750,0.900],                                    
                                    7:[0.010,0.100,0.250,0.500,0.750,0.900,0.990],                                    
                                    8:[0.001,0.020,0.100,0.250,0.500,0.750,0.900,0.990],                                    
                                    9:[0.001,0.020,0.100,0.250,0.500,0.750,0.900,0.980,0.990],                                    
                                    10:[0.001,0.010,0.050,0.100,0.250,0.500,0.750,0.900,0.950,0.999],                                    
                                    11:[0.001,0.010,0.050,0.100,0.250,0.500,0.660,0.750,0.900,0.950,0.999],                                    
                                    12:[0.001,0.010,0.050,0.100,0.250,0.350,0.500,0.660,0.750,0.900,0.950,0.999],                                    
                                    13:[0.001,0.010,0.050,0.100,0.250,0.350,0.500,0.660,0.750,0.900,0.950,0.980,0.999],                                    
                                    14:[0.001,0.010,0.030,0.050,0.100,0.250,0.350,0.500,0.650,0.750,0.800,0.900,0.950,0.999],                                    
                                    15:[0.001,0.005,0.010,0.050,0.100,0.250,0.350,0.500,0.650,0.750,0.900,0.950,0.970,0.990,0.999],                                    
                                    16:[0.001,0.005,0.010,0.050,0.100,0.250,0.350,0.500,0.650,0.750,0.900,0.950,0.970,0.990,0.995,0.999]                                    
                                } 
    #add SPT normal or three terms
    number_of_quantiles = int(st.sidebar.slider('Number of Quantiles',3,16,key='Quantile'))
    quanile_container = st.sidebar.container()
    quanile_container.subheader("Please Enter Values Below:")
    y_values, x_values = quanile_container.columns(2)
    q_data = []
    for num in range(1,number_of_quantiles+1):
        q_data.append([float(y_values.text_input(f'Percentage {num}',reference_probabilities[number_of_quantiles][num - 1] ,key=f"y values {num}")),
                        float(x_values.text_input(f'Value {num}', '0',key=f"x values {num}"))])
        # if num > 1 and any(q_data[-1]):
            # quanile_container.error(f"Please enter a number greater zero for Value {num}.")
        # Add check that items are less than the other value percentage
        # if len(q_data) > 1:
            # q_data
    pd_data = pd.DataFrame(q_data,columns=['y','x'])
    # st.subheader("Preview of Quantile Data")
    # st.write(pd_data.to_html(index=False), unsafe_allow_html=True)
    boundedness = quanile_container.selectbox('Graph Boundedness', ("'u' - unbounded", 
                                                       "'sl' - semi-bounded lower", 
                                                       "'su' - semi-bounded upper",
                                                       "'b' - bounded on both sides"),key=f"Quantile_boundedness")
                                                       
    if boundedness == "'b' - bounded on both sides":
        #convert to float and list
        boundsl = y_values.text_input('Lower Bound', '0',key=f"Quantile_lower")
        boundsu = x_values.text_input('Upper Bound', '1',key=f"Quantile_upper")
        bounds = [float(boundsl),float(boundsu)]
    elif boundedness.find("lower") != -1:
        bounds = [float(y_values.text_input('Lower Bound', '0',key=f"Quantile_lower"))]
    elif boundedness.find("upper") != -1:
        bounds = [float(x_values.text_input('Upper Bound', '1',key=f"Quantile_upper"))]
    else:
        bounds = [0,1]
        
    boundedness = boundedness.strip().split(" - ")[0].replace("'","")
	
    if quanile_container.button("Make Graphs"):
        preprocess_charts(pd_data['x'].to_list(),pd_data['y'].to_list(),boundedness,bounds)
        # pass
    input_data("Unknown",0,pd_data[['x']],pd_data['y'].to_list())