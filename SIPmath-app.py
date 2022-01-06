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
import warnings
warnings.filterwarnings('ignore')

PM_logo = Image.open('images\PM_logo.png')
Metalog_Distribution = Image.open('images\Metalog Distribution.png')
HDR_Generator = Image.open('images\HDR Generator.png')
SIPmath_Standard = Image.open('images\SIPmath Standard.png')
# image = Image.open('PM_logo_transparent.png')
st.set_page_config(page_title="SIPmath™ 3.0 Library Generator", page_icon=None, layout="wide", initial_sidebar_state="auto", menu_items=None)
images_container = st.container()
images_cols = images_container.columns([0.75,1.25,0.3,0.25,0.3])
images_cols[0].markdown("##### [![Probability Management](https://images.squarespace-cdn.com/content/v1/5a4f82d7a8b2b04080732f87/1590000132586-Q3BM496CR13EESETZTR6/PM_logo_transparent.png?format=150w)](https://www.probabilitymanagement.org/)")
images_cols[4].markdown("##### [![Metalog Distribution](https://images.squarespace-cdn.com/content/v1/5a4f82d7a8b2b04080732f87/1619026311825-CR5I84A3ZD58DNE604NC/Metalogs.png?format=100w)](https://www.probabilitymanagement.org/metalog)  Metalog Distribution")
images_cols[1].header("SIPmath™ 3.0 Library Generator")
images_cols[3].markdown("##### [![HDR Generator](https://images.squarespace-cdn.com/content/v1/5a4f82d7a8b2b04080732f87/1619038789110-RGU809AE2A6JFLVVHTH4/HDR.png)](https://www.probabilitymanagement.org/hdr) HDR Generator")
images_cols[2].markdown("##### [![SIPmath Standard](https://images.squarespace-cdn.com/content/v1/5a4f82d7a8b2b04080732f87/1631227023221-SOKANKYGE29JXUD3PB7E/SIPmath+Standard+reversed.png?format=100w)](https://www.probabilitymanagement.org/30-standard) SIPmath Standard")
# images_container
# images_container.image(image,width=1000)
main_container = st.empty()
empty_table = st.empty()
table_container = empty_table.container()
slider_container = st.container()
graphs_container = st.empty().container()
#Taken from the metalog
# @st.cache(suppress_st_warning=True)
def plot(m, big_plots=None,csv=None,term=None,name=None):
    
    if csv:
      key = 'csv'
    else:
      key = 'quantile'
    # Collecting data to set limits of axes
    res_data = pd.DataFrame({'term': np.repeat(str(m['params']['term_lower_bound']) \
                                                     + ' Terms', len(m['M'].iloc[:, 0])),
                                   'pdfValues': m['M'].iloc[:, 0],
                                   'quantileValues': m['M'].iloc[:, 1],
                                   'cumValue': m['M']['y']
                                   })
    if m['M'].shape[-1] > 3:
        for i in range(2, len(m['M'].iloc[0, ] - 1) // 2 + 1):
            if m['Validation']['valid'][i] == 'yes':
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
    max_valid_term = m['Validation'][m['Validation']['valid'] == 'yes']['term'].max()
    
    results_len = len(InitialResults)
    # fig, ax = plt.subplots(results_len, 2, figsize=(8, 3*results_len), sharex='col')
    if big_plots:
        fig, ax = plt.subplots(1, 2, figsize=(10, 3), sharex='col')    
        # if st.session_state['mfitted'][key][name]['fit']
        # fig, ax = plt.subplots(1, 2, figsize=(4, 2), sharex='col')        
        # i = 2
        if term is None:
            for i in range(results_len+1,1,-1):
                if m['Validation']['valid'][i] == 'yes':
                    j = 0
                    # Plotting PDF
                    ax[j].plot(InitialResults[str(i) + ' Terms']['quantileValues'], InitialResults[str(i) + ' Terms']['pdfValues'],
                          linewidth=2, label=str(i) + ' Terms')
                    # Plot data 
                    ax[j + 1].scatter(m["dataValues"]['x'],m["dataValues"]['probs'],c='white',edgecolor='black', label=f'{name} Data')
                    # Plotting CDF
                    ax[j + 1].plot(InitialResults[str(i) + ' Terms']['quantileValues'], InitialResults[str(i) + ' Terms']['cumValue'],
                          linewidth=2, label=str(i) + ' Terms')
                    ax[j].patch.set_facecolor('white')
                    # ax[j].axes.xaxis.set_ticks([])     
                    # ax[j].axes.yaxis.set_ticks([])     
                    ax[j + 1].patch.set_facecolor('white')
                    ax[j + 1].axes.xaxis.set_ticks([])     
                    # ax[j + 1].axes.yaxis.set_ticks([])     
                    # ax[j].set(title=str(i) + ' Terms', ylabel='PDF', xlabel='Quantiles')
                    # ax[j + 1].set(title=str(i) + ' Terms', ylabel='CDF', xlabel='Quantiles')
                    ax[j].set(title=str(i) + ' Terms', ylabel='PDF')
                    ax[j + 1].set(title=str(i) + ' Terms', ylabel='CDF')
                    ax[j].axis([min(res_data['quantileValues']), max(res_data['quantileValues']), 0, max(res_data["pdfValues"])]) 
                    ax[j+1].axis([min(res_data['quantileValues']), max(res_data['quantileValues']), round(min(m["dataValues"]['probs']),1), round(max(m["dataValues"]['probs']),1)]) 
                    # if 'big plots' not in st.session_state['mfitted'][key][name]['plot']:
                        # st.session_state['mfitted'][key][name]['plot']['big plot'] = plt 
                        # return                        
                    break
        else:
            terms_for_loop = [term, max_valid_term]
            # for i in range(2,term+1):
            for i in terms_for_loop :
                if m['Validation']['valid'][i] == 'yes':
                    j = 0
                    # Plotting PDF
                    ax[j].plot(InitialResults[str(i) + ' Terms']['quantileValues'], InitialResults[str(i) + ' Terms']['pdfValues'],
                          linewidth=2,label=f'{i} Terms')
                    # Plot data 
                    ax[j + 1].scatter(m["dataValues"]['x'],m["dataValues"]['probs'],c='white',edgecolor='black')
                    # Plotting CDF
                    ax[j + 1].plot(InitialResults[str(i) + ' Terms']['quantileValues'], InitialResults[str(i) + ' Terms']['cumValue'],
                          linewidth=2)
                    ax[j].patch.set_facecolor('white')
                    # ax[j].axes.xaxis.set_ticks([])     
                    ax[j].axes.yaxis.set_ticks([])     
                    ax[j + 1].patch.set_facecolor('white')
                    # ax[j + 1].axes.xaxis.set_ticks([])     
                    ax[j + 1].axes.yaxis.set_ticks([])     
                    # ax[j].legend(loc='upper center', bbox_to_anchor=(1.05, 0.05), fancybox=True, shadow=True,ncol=2)
                    # ax[j].legend(loc='upper center',ncol=2)
                    # ax[j + 1].legend([str(i) + ' Terms'])
                    # ax[j].axis([min(res_data['quantileValues']), max(res_data['quantileValues']), 0, max(res_data["pdfValues"])]) 
                    ax[j+1].axis([min(res_data['quantileValues']), max(res_data['quantileValues']), round(min(m["dataValues"]['probs']),1), round(max(m["dataValues"]['probs']),1)]) 
                    # ax[j].set(title=str(i) + ' Terms', ylabel='PDF', xlabel='Quantiles')
                    # ax[j + 1].set(title=str(i) + ' Terms', ylabel='CDF', xlabel='Quantiles')
            if len(terms_for_loop) == 2:
                chart_title = " and ".join([str(x) for x in terms_for_loop if m['Validation']['valid'][x] == 'yes']) + ' Terms'
                ax[0].set(title=chart_title, ylabel='PDF', xlabel='Quantiles')
                ax[1].set(title=chart_title, ylabel='CDF', xlabel='Quantiles')
            else:
                ax[0].set(title=str(i) + ' Terms', ylabel='PDF')
                ax[1].set(title=str(i) + ' Terms', ylabel='CDF')
            if 'big plots' not in st.session_state['mfitted'][key][name]['plot']:
                st.session_state['mfitted'][key][name]['plot']['big plot'] = plt
                # return
                    # break
        # ax[0].legend()
        plt.tight_layout(rect=[0,0,0.75,1])
        graphs_container.pyplot(plt)
        
        # plt.clf()
                       # ax[i-2, j].patch.set(title=str(current_term) + ' Terms', ylabel='PDF', xlabel='Quantiles')
                          
                              
                    # if current_term != 5*3:
                        # ax[i-2, j].set(title=str(current_term) + ' Terms', ylabel='CDF')
                    # else:
                       # ax[i-2, j].set(title=str(current_term) + ' Terms', ylabel='CDF', xlabel='Quantiles')fig, ax = plt.subplots(5, 3, figsize=(8, 3*3), sharex='col')
                       
        # i = 2
    if csv is False:  
        fig, ax = plt.subplots(5, 3, figsize=(8, 3*3), sharex='col')
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
        graphs_container.pyplot(plt)
    # return plt

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
                                       bounds,
                                       big_plots,
                                       terms,
                                       csv,
                                       name):
	#Create metalog
	# st.write(boundedness,
                       # bounds)
	if 'mfitted' not in st.session_state:
	  st.session_state['mfitted'] = {'csv':{},'quantile':{}}
	if csv:
	  key = 'csv'
	else:
	  key = 'quantile'
	if 'mfitted' not in st.session_state['mfitted'][key]:
	  mfitted = metalog.fit(x, bounds = bounds, boundedness = boundedness, fit_method='OLS', term_limit = terms, probs=probs)
	  st.session_state['mfitted'][key]= {name:{'fit':mfitted,'plot':{'csv':None,'big plot':None}}}
	  max_valid_term = mfitted['Validation'][mfitted['Validation']['valid'] == 'yes']['term'].max()
    
    #Create graphs
	st.write(st.session_state['mfitted'][key][name]['fit'].keys())
	st.write(type(st.session_state['mfitted'][key][name]['fit']))
	st.write(st.session_state['mfitted'][key][name]['fit']['M'])
	st.write(st.session_state['mfitted'][key][name]['fit']["dataValues"])
	st.write(st.session_state['mfitted'][key][name]['fit']['Validation'])
    # big_plots = st.sidebar.checkbox("Big Graphs?")
	max_valid_term = st.session_state['mfitted'][key][name]['fit']['Validation'][st.session_state['mfitted'][key][name]['fit']['Validation']['valid'] == 'yes']['term'].max()
	if big_plots:
	    term = graphs_container.slider(f"Select {name} Terms: ",2,max_valid_term,key=f"{name} term slider")
	else:
	    term = 16
	# plot(mfitted, True,csv=None,term=None,name=name)
	plot(st.session_state['mfitted'][key][name]['fit'],big_plots,csv,term,name=name)

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
 
def make_csv_graph(series,
                                       probs,
                                       boundedness,
                                       bounds,
                                       big_plots):
    if big_plots:
        graphs_container.header(series.name)
    preprocess_charts(series.to_list(),
                                    probs,
                                    boundedness,
                                    bounds,
                                    big_plots,
                                    16,
                                    True,
                                    series.name)
                                    
    return None

def input_data(name,i,df,probs=None):
    if probs is None:
        probs = np.nan
        max_val = 16
        default_val = 5
    else:
        max_val = df.shape[0]
        default_val = max_val
    table_container.write("If the data above appears correct, please enter your parameters in the sidebar for this file.")
    with st.sidebar.expander("See JSON Options"):
        filename = st.text_input(f'Filename {i+1}', name+'.SIPmath',key=f"{name}_{i}_filename")
        author = st.text_input(f'Author for {filename}', 'Unknown',key=f"{name}_author")
        dependence = st.selectbox('Dependence', ('independent','dependent'),key=f"{name}_{i}_dependence")
        boundedness = st.selectbox('Boundedness', ("'u' - unbounded", 
                                                           "'sl' - semi-bounded lower", 
                                                           "'su' - semi-bounded upper",
                                                           "'b' - bounded on both sides"),key=f"{name}_boundedness")
                                                           
        if boundedness == "'b' - bounded on both sides":
            #convert to float and list
            boundsl = st.text_input('Lower Bound', '0',key=f"{name}_lower")
            boundsu = st.text_input('Upper Bound', '1',key=f"{name}_upper")
            bounds = [float(boundsl),float(boundsu)]
        elif boundedness.find("lower") != -1:
            bounds = [float(st.text_input('Lower Bound', '0',key=f"{name}_lower"))]
        elif boundedness.find("upper") != -1:
            bounds = [float(st.text_input('Upper Bound', '1',key=f"{name}_upper"))]
        else:
            bounds = [0,1]
            
        boundedness = boundedness.strip().split(" - ")[0].replace("'","")
        if max_val > 3:
            term_saved = st.slider('Term Saved',3,max_val,default_val,key=f"{name}_term_saved")
        else:
             term_saved = 3

        if st.button(f'Convert to {name} SIPmath Json?',key=f"{name}_term_saved"):
            graphs_container.empty()
            table_container.subheader("Preview and Download the JSON file below.")
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
            table_container.text("Mouse over the text then click on the clipboard icon to copy to your clipboard.")
            with open(filename, 'rb') as f:
                table_container.json(json.load(f))

#st.title('SIPmath JSON Creator')
st.sidebar.header('User Input Parameters')

# Collects user input features into dataframe
data_type = st.sidebar.radio('Input Data Type', ('CSV File','Quantile'), index=0)

if data_type == 'CSV File':
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"],accept_multiple_files=True)
    if uploaded_file != None:
        input_df = None
        for i,file in enumerate(uploaded_file):
            try:
                input_df = pd.read_csv(file)
            except UnicodeDecodeError:
                input_df = pd.read_csv(file,encoding='cp437')
            # [for x in i]
            name = file.name.replace(".csv","")
            # with main_contanier.contanier():
            table_container.subheader(f"Preview for {name}")
            # table_container.write(input_df[:10].to_html(index=False), unsafe_allow_html=True)
        if isinstance(input_df, pd.DataFrame):
            quanile_container = st.sidebar.container()
            # quanile_container.subheader("Please Enter Values Below:")
            y_values, x_values = quanile_container.columns(2)
            boundedness = quanile_container.selectbox('Boundedness', ("'u' - unbounded", 
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
            big_plots = quanile_container.checkbox("Big Graphs?",value=False)
            make_graphs_checkbox = quanile_container.button("Make Graphs")
            if make_graphs_checkbox or all(input_df.any()):
                if big_plots:
                    # if not "selected_column" in locals():
                        # empty_table.empty()
                    selected_column = graphs_container.selectbox("Select Column:", input_df.columns,key="Big Graph Column")
                    input_df[[selected_column]].apply(make_csv_graph,
                                                   probs = np.nan,
                                                   boundedness = boundedness,
                                                   bounds = bounds,
                                                   big_plots = big_plots)
                # input_df.apply(make_csv_graph,
                                               # probs = np.nan,
                                               # boundedness = boundedness,
                                               # bounds = bounds,
                                               # big_plots = big_plots)
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
    boundedness = quanile_container.selectbox('Boundedness', ("'u' - unbounded", 
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
    big_plots = quanile_container.checkbox("Big Graphs?")
    if quanile_container.button("Make Graphs") or all(pd_data.any()):
        preprocess_charts(pd_data['x'].to_list(),
                                        pd_data['y'].to_list(),
                                        boundedness,
                                        bounds,
                                        big_plots,
                                        pd_data.shape[0],
                                        False,
                                        pd_data['x'].name)
        # pass
    input_data("Unknown",0,pd_data[['x']],pd_data['y'].to_list())