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
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
warnings.filterwarnings('ignore')

PM_logo = Image.open('images\PM_logo.png')
Metalog_Distribution = Image.open('images\Metalog Distribution.png')
HDR_Generator = Image.open('images\HDR Generator.png')
SIPmath_Standard = Image.open('images\SIPmath Standard.png')
# image = Image.open('PM_logo_transparent.png')
st.set_page_config(page_title="SIPmath™ 3.0 Library Generator", page_icon=None, layout="wide", initial_sidebar_state="auto", menu_items=None)
images_container = st.container()
images_cols = images_container.columns([0.75,1.25,0.3,0.25,0.3])
# images_cols[0].markdown("##### [![Probability Management](https://images.squarespace-cdn.com/content/v1/5a4f82d7a8b2b04080732f87/1590000132586-Q3BM496CR13EESETZTR6/PM_logo_transparent.png?format=150w)](https://www.probabilitymanagement.org/)")
images_cols[0].image(PM_logo,width=300)
images_cols[4].image(Metalog_Distribution,width=120)
images_cols[1].header("SIPmath™ 3.0 Library Generator")
images_cols[3].image(HDR_Generator,width=110)
images_cols[2].image(SIPmath_Standard,width=105)
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
    # st.write(m)
    if csv:
      key = 'csv'
    else: 
      key = 'quantile'
    # if res_data
    # Collecting data to set limits of axes
    if 'res_data' not in st.session_state['mfitted'][key][name]:
        # st.write("notthere")
        res_data = pd.DataFrame({'term': np.repeat(str(m['params']['term_lower_bound']) \
                                                         + ' Terms', len(m['M'].iloc[:, 0])),
                                       'pdfValues': m['M'].iloc[:, 0],
                                       'quantileValues': m['M'].iloc[:, 1],
                                       'cumValue': m['M']['y']
                                       })
        if m['M'].shape[-1] > 3:
            for i in range(2, len(m['M'].iloc[0, ] - 1) // 2 + 1):
                if m['Validation']['valid'][i] == 'yes':
                    # st.write(i)
                    temp_data = pd.DataFrame({'term': np.repeat(str(m['params']['term_lower_bound'] + i - 1) \
                                                                  + ' Terms', len(m['M'].iloc[:, 0])),
                                                'pdfValues': m['M'].iloc[:, i * 2 - 2],
                                                'quantileValues': m['M'].iloc[:, i * 2 - 1],
                                                'cumValue': m['M']['y']})
                    res_data = pd.concat([res_data, temp_data], ignore_index=True)
        res_data['frames'] =  res_data['term']
        res_data['groups'] = res_data['term']
    else:
        res_data = st.session_state['mfitted'][key][name]['res_data']
    # st.write(res_data)
    # st.write(m['Validation']['valid'])
    if (res_data['term'] != f"{term} Terms").all():
      for new in range(int(term),1,-1):
        # st.write(new)
        if (res_data['term'] == f"{new} Terms").any():
          term = new
          # st.write(new)
          break
    highest_term = f"{term} Terms"
    # print(term)
    # st.write(highest_term)
    highest_term_df = res_data[res_data['term'] == highest_term]
    # st.write(highest_term_df)
    highest_term_index = highest_term_df.index[-1] + 1
    # highest_term_df['frames']
    fig = px.line(res_data[:highest_term_index], y="pdfValues", x="quantileValues", color = "term",
              animation_group='groups',
              animation_frame = 'frames',
              range_x=[min(res_data[:highest_term_index]['quantileValues']), max(res_data[:highest_term_index]['quantileValues'])],
              range_y =[0, max(res_data[:highest_term_index]["pdfValues"])])
    fig1 = px.line(res_data[:highest_term_index], y="cumValue", x="quantileValues", color = "term",
              animation_group='term',
              animation_frame = 'term')
    # st.write(res_data)# hide and lock down axes
    # fig.update_xaxes(visible=False, fixedrange=True)
    # fig.update_yaxes(visible=False, fixedrange=True)
    # fig.for_each_trace(
                                # lambda trace: trace.update(visible = "legendonly") if trace.name != highest_term  else (),
        # )
    # fig1.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 750
    fig.update_layout(
                   showlegend=False,
                   yaxis_title='PDF',
                   xaxis_title='Quantile',
                   paper_bgcolor='rgba(0,0,0,0)',
                   plot_bgcolor='rgba(0,0,0,0)',
                   xaxis=dict(
                      showline=True,
                      showgrid=False,
                      showticklabels=True,
                      linecolor='rgb(204, 204, 204)',
                      linewidth=2,
                      ticks='outside',
                      tickfont=dict(
                          family='Arial',
                          size=12,
                          color='rgb(82, 82, 82)',
                      ),
                  ),
                   yaxis=dict(
                      showgrid=False,
                      zeroline=False,
                      showline=False,
                      showticklabels=False,
                  ))
    fig.add_trace(go.Scatter(y=highest_term_df.pdfValues, x=highest_term_df.quantileValues,name=highest_term))
    fig1.add_trace(go.Scatter(mode='markers', y=m["dataValues"]['probs'], x=m["dataValues"]['x'],
        marker=dict(
            color='rgba(0,0,0,0)',
            size=15,
            line=dict(
            color='DarkRed',
                width=1
            )),name=name))
    fig1.add_trace(go.Scatter(y=highest_term_df.cumValue, x=highest_term_df.quantileValues,name=highest_term))
    total_graph = make_subplots(rows=1, cols=2, subplot_titles = ('PDF', 'CDF'))
    _ = [total_graph.append_trace(trace, row=1, col=1) for trace in fig['data']]
    _ = [total_graph.append_trace(trace, row=1, col=2) for trace in fig1['data']]
    total_graph.update_layout(
                   showlegend=False,
                   paper_bgcolor='rgba(0,0,0,0)',
                   plot_bgcolor='rgba(0,0,0,0)',
                   xaxis=dict(
                      range=[min(res_data[:highest_term_index]['quantileValues']), max(res_data[:highest_term_index]['quantileValues'])],
                      showline=True,
                      showgrid=False,
                      showticklabels=True,
                      linecolor='rgb(204, 204, 204)',
                      linewidth=2,
                      ticks='outside',
                      tickfont=dict(
                          family='Arial',
                          size=12,
                          color='rgb(82, 82, 82)',
                      ),
                  ),
                   xaxis2=dict(
                      range=[min(res_data[:highest_term_index]['quantileValues']), max(res_data[:highest_term_index]['quantileValues'])],
                      showline=True,
                      showgrid=False,
                      showticklabels=True,
                      linecolor='rgb(204, 204, 204)',
                      linewidth=2,
                      ticks='outside',
                      tickfont=dict(
                          family='Arial',
                          size=12,
                          color='rgb(82, 82, 82)',
                      ),
                  ),
                   yaxis=dict(
                      showgrid=False,
                      zeroline=False,
                      showline=False,
                      showticklabels=False,
                      range = [0, max(res_data[:highest_term_index]["pdfValues"])]
                  ),
                   yaxis2=dict(
                      showgrid=False,
                      zeroline=False,
                      showline=False,
                      showticklabels=False,
                      range = [0, max(res_data[:highest_term_index]["cumValue"])]
                  ))
    # total_graph.update_yaxes(range=[0,1])
    frames = [dict(
                   name = k,
                   data = [go.Scatter(y= res_data.loc[res_data['term'] == f"{k} Terms",'pdfValues'], x=res_data.loc[res_data['term'] == f"{k} Terms",'quantileValues']),#update the trace 1 in (1,1)
                               # go.Scatter(y=highest_term_df.pdfValues),#update the trace 1 in (1,1)
                               go.Scatter(y = res_data.loc[res_data['term'] == f"{k} Terms",'cumValue'], x=res_data.loc[res_data['term'] == f"{k} Terms",'quantileValues']),
                               # go.Scatter(y=m["dataValues"]['probs']),
                               # go.Scatter(y=highest_term_df.cumValue)
                           ],
                   traces=[0,2]# the elements of the list [0,1,2] give info on the traces in fig.data
                                          # that are updated by the above three go.Scatter instances
                  ) for k in range(term+1) if (res_data['term'] == f"{k} Terms").any()]
    # st.write(total_graph.layout)
    updatemenus = [dict(type='buttons',
                                  buttons=[dict(label='Play',
                                  method='animate',
                                  args=[[f'{k}' for k in range(term+1) if (res_data['term'] == f"{k} Terms").any()], 
                                         dict(frame=dict(duration=750, redraw=True), 
                                              transition=dict(duration=250),
                                              easing='linear',
                                              fromcurrent=True,
                                              mode='immediate'
                                                                 )])],
                    direction= 'left', 
                    pad=dict(r= 10, t=85), 
                    showactive =True, x= 0.1, y= 0, xanchor= 'right', yanchor= 'top')
            ]

    sliders = [{'yanchor': 'top',
                'xanchor': 'left', 
                'currentvalue': {'font': {'size': 16}, 'prefix': 'Term: ', 'visible': True, 'xanchor': 'right'},
                'transition': {'duration': 500.0, 'easing': 'linear'},
                'pad': {'b': 10, 't': 50}, 
                'len': 0.9, 'x': 0.1, 'y': 0, 
                'steps': [{'args': [[k], {'frame': {'duration': 500.0, 'easing': 'linear', 'redraw': False},
                                          'transition': {'duration': 100, 'easing': 'linear'}}], 
                           'label': k, 'method': 'animate'} for k in range(term+1)  if (res_data['term'] == f"{k} Terms").any()     
                        ]}]                       
    total_graph.update(frames=frames)
    total_graph.update_layout(updatemenus=updatemenus,
                  sliders=sliders)
    graphs_container.plotly_chart(total_graph, use_container_width=True)
    
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
    
    # st.write(m['M'])
    # st.write(InitialResults)
    
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
            # terms_for_loop = [term, max_valid_term]
            terms_for_loop = [term]
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
                        # chart_title = " and ".join([str(x) for x in terms_for_loop if m['Validation']['valid'][x] == 'yes']) + ' Terms'
                        chart_title = f'{i} Terms'
                        ax[0].set(title=chart_title, ylabel='PDF', xlabel='Quantiles')
                        ax[1].set(title=chart_title, ylabel='CDF', xlabel='Quantiles')
                    else:
                        ax[0].set(title=str(i) + ' Terms', ylabel='PDF')
                        ax[1].set(title=str(i) + ' Terms', ylabel='CDF')
                else:
                    ax[0].patch.set_facecolor('white')
                    # ax[j].axes.xaxis.set_ticks([])     
                    ax[0].axes.yaxis.set_ticks([])     
                    ax[1].patch.set_facecolor('white')
                    # ax[j + 1].axes.xaxis.set_ticks([])     
                    ax[1].axes.yaxis.set_ticks([])     
                    chart_title = f'{term} Terms'
                    ax[0].set(title=chart_title, ylabel='PDF', xlabel='Quantiles')
                    ax[1].set(title=chart_title, ylabel='CDF', xlabel='Quantiles')
                    
            if 'big plots' not in st.session_state['mfitted'][key][name]['plot']:
                st.session_state['mfitted'][key][name]['plot']['big plot'] = plt
                # return
                    # break
        # ax[0].legend()
        plt.tight_layout(rect=[0,0,0.75,1])
        # graphs_container.pyplot(plt)
        
        # plt.clf()
                       # ax[i-2, j].patch.set(title=str(current_term) + ' Terms', ylabel='PDF', xlabel='Quantiles')
                          
                              
                    # if current_term != 5*3:
                        # ax[i-2, j].set(title=str(current_term) + ' Terms', ylabel='CDF')
                    # else:
                       # ax[i-2, j].set(title=str(current_term) + ' Terms', ylabel='CDF', xlabel='Quantiles')fig, ax = plt.subplots(5, 3, figsize=(8, 3*3), sharex='col')
                       
        # i = 2
    # if csv is False:  
    fig, ax = plt.subplots(3, 5, figsize=(10, 5), sharex='col')
    for i in range(2, 4 + 1):
        for j in range(0,5):
            current_term = (2 + (i - 2)*5 + j) 
            print(f"{current_term}")
            if results_len + 2 > current_term and m['Validation']['valid'][current_term] == 'yes':# Check to make sure it is valid before plotting.
                print(f"plotting {current_term}")
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
            if current_term < 11:
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
    # graphs_container.pyplot(plt)
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
	# st.write(st.session_state['mfitted'][key][name]['fit'].keys())
	# st.write(type(st.session_state['mfitted'][key][name]['fit']))
	# st.write(st.session_state['mfitted'][key][name]['fit']['M'])
	# st.write(st.session_state['mfitted'][key][name]['fit']["dataValues"])
	# st.write(st.session_state['mfitted'][key][name]['fit']['Validation'])
    # big_plots = st.sidebar.checkbox("Big Graphs?")
	max_valid_term = int(st.session_state['mfitted'][key][name]['fit']['Validation'][st.session_state['mfitted'][key][name]['fit']['Validation']['valid'] == 'yes']['term'].max())
	# print(type(max_valid_term))
	if big_plots:
	    term = graphs_container.slider(f"Select {name} Terms: ",int(3),max_valid_term,key=f"{name} term slider")
	else:
	    term = int(16)
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
# @st.cache
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
            table_container.write(input_df[:10].to_html(index=False), unsafe_allow_html=True)
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