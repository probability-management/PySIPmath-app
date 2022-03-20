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
import os
import io
import time
import copy as cp
import altair as alt
import base64
warnings.filterwarnings('ignore')
st.set_page_config(page_title="SIPmath™ 3.0 Library Generator", page_icon=None, layout="wide", initial_sidebar_state="auto", menu_items=None)
@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

@st.cache(allow_output_mutation=True)
def get_img_with_href(local_img_path, target_url):
    img_format = os.path.splitext(local_img_path)[-1].replace('.', '')
    bin_str = get_base64_of_bin_file(local_img_path)
    html_code = f'''
        <a href="{target_url}">
            <img src="data:image/{img_format};base64,{bin_str}" height="125"; />
        </a>'''
    return html_code
path = os.path.dirname(__file__)
# PM_logo = Image.open(path+'/images/PM_logo.png')
# Metalog_Distribution = Image.open(path+'/images/Metalog Distribution.png')
# HDR_Generator = Image.open(path+'/images/HDR Generator.png')
# SIPmath_Standard = Image.open(path+'/images/SIPmath Standard.png')
PM_logo = get_img_with_href(path+'/images/PM_logo.png','https://www.probabilitymanagement.org/')
Metalog_Distribution = get_img_with_href(path+'/images/Metalog Distribution.png','https://www.probabilitymanagement.org/metalog')
HDR_Generator = get_img_with_href(path+'/images/HDR Generator.png','https://www.probabilitymanagement.org/hdr')
SIPmath_Standard = get_img_with_href(path+'/images/SIPmath Standard.png','https://www.probabilitymanagement.org/sipmath')
# image = Image.open('PM_logo_transparent.png')
images_container = st.container()
images_cols = images_container.columns([13,11,3,3,3])
# images_cols[0].markdown("##### [![Probability Management](https://images.squarespace-cdn.com/content/v1/5a4f82d7a8b2b04080732f87/1590000132586-Q3BM496CR13EESETZTR6/PM_logo_transparent.png?format=150w)](https://www.probabilitymanagement.org/)")
# images_cols[0].image(PM_logo, unsafe_allow_html=True)
# images_cols[4].image(Metalog_Distribution, unsafe_allow_html=True)
# images_cols[1].header("SIPmath™ 3.0 Library Generator")
# images_cols[3].image(HDR_Generator, unsafe_allow_html=True)
# images_cols[2].image(SIPmath_Standard, unsafe_allow_html=True)
images_cols[0].markdown(PM_logo, unsafe_allow_html=True)
images_cols[4].markdown(Metalog_Distribution, unsafe_allow_html=True)
images_cols[1].header("SIPmath™ 3.0 Library Generator")
images_cols[3].markdown(HDR_Generator, unsafe_allow_html=True)
images_cols[2].markdown(SIPmath_Standard, unsafe_allow_html=True)
# images_container
# images_container.image(image, unsafe_allow_html=True)
main_container = st.empty()
empty_table = st.empty()
table_container = empty_table.container()
slider_container = st.empty().container()
graphs_container = st.empty().container()
graphs_container_main = st.empty().container()
#Taken from the metalog
# @st.cache(suppress_st_warning=True)

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 500px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 500px;
        margin-left: -500px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

def plot(m, big_plots=None,csv=None,term=None,name=None,key=None):
    # st.write(m)
    # print("is_quantile is ",is_quantile," csv is ",csv)
    # if is_quantile or not csv: 
      # key = 'quantile'
    # else:
      # key = 'csv'
    # if res_data
    # Collecting data to set limits of axes
    print(f"running plot for {name} in {key}")
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
        st.session_state['mfitted'][key][name]['res_data'] = res_data
    else:
        res_data = st.session_state['mfitted'][key][name]['res_data']
    # st.write(res_data)
    # st.write(m['Validation']['valid'])
    # if (res_data['term'] != f"{term} Terms").all():
      # for new in range(int(term),1,-1):
        # # st.write(new)
        # if (res_data['term'] == f"{new} Terms").any():
          # term = new
          # # st.write(new)
          # break
    # highest_term = f"{term} Terms"
    # # print(term)
    # # st.write(highest_term)
    # highest_term_df = res_data[res_data['term'] == highest_term]
    # # st.write(highest_term_df)
    # highest_term_index = highest_term_df.index[-1] + 1
    # # highest_term_df['frames']
    # fig = px.line(res_data[:highest_term_index], y="pdfValues", x="quantileValues", color = "term",
              # animation_group='groups',
              # animation_frame = 'frames',
              # range_x=[min(res_data[:highest_term_index]['quantileValues']), max(res_data[:highest_term_index]['quantileValues'])],
              # range_y =[0, max(res_data[:highest_term_index]["pdfValues"])])
    # fig1 = px.line(res_data[:highest_term_index], y="cumValue", x="quantileValues", color = "term",
              # animation_group='term',
              # animation_frame = 'term')
    # # st.write(res_data)# hide and lock down axes
    # # fig.update_xaxes(visible=False, fixedrange=True)
    # # fig.update_yaxes(visible=False, fixedrange=True)
    # # fig.for_each_trace(
                                # # lambda trace: trace.update(visible = "legendonly") if trace.name != highest_term  else (),
        # # )
    # # fig1.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 750
    # fig.update_layout(
                   # showlegend=False,
                   # yaxis_title='PDF',
                   # xaxis_title='Quantile',
                   # paper_bgcolor='rgba(0,0,0,0)',
                   # plot_bgcolor='rgba(0,0,0,0)',
                   # xaxis=dict(
                      # showline=True,
                      # showgrid=False,
                      # showticklabels=True,
                      # linecolor='rgb(204, 204, 204)',
                      # linewidth=2,
                      # ticks='outside',
                      # tickfont=dict(
                          # family='Arial',
                          # size=12,
                          # color='rgb(82, 82, 82)',
                      # ),
                  # ),
                   # yaxis=dict(
                      # showgrid=False,
                      # zeroline=False,
                      # showline=False,
                      # showticklabels=False,
                  # ))
    # fig.add_trace(go.Scatter(y=highest_term_df.pdfValues, x=highest_term_df.quantileValues,name=highest_term))
    # fig1.add_trace(go.Scatter(mode='markers', y=m["dataValues"]['probs'], x=m["dataValues"]['x'],
        # marker=dict(
            # color='rgba(0,0,0,0)',
            # size=15,
            # line=dict(
            # color='DarkRed',
                # width=1
            # )),name=name))
    # fig1.add_trace(go.Scatter(y=highest_term_df.cumValue, x=highest_term_df.quantileValues,name=highest_term))
    # total_graph = make_subplots(rows=1, cols=2, subplot_titles = ('PDF', 'CDF'))
    # _ = [total_graph.append_trace(trace, row=1, col=1) for trace in fig['data']]
    # _ = [total_graph.append_trace(trace, row=1, col=2) for trace in fig1['data']]
    # total_graph.update_layout(
                   # showlegend=False,
                   # paper_bgcolor='rgba(0,0,0,0)',
                   # plot_bgcolor='rgba(0,0,0,0)',
                   # xaxis=dict(
                      # range=[min(res_data[:highest_term_index]['quantileValues']), max(res_data[:highest_term_index]['quantileValues'])],
                      # showline=True,
                      # showgrid=False,
                      # showticklabels=True,
                      # linecolor='rgb(204, 204, 204)',
                      # linewidth=2,
                      # ticks='outside',
                      # tickfont=dict(
                          # family='Arial',
                          # size=12,
                          # color='rgb(82, 82, 82)',
                      # ),
                  # ),
                   # xaxis2=dict(
                      # range=[min(res_data[:highest_term_index]['quantileValues']), max(res_data[:highest_term_index]['quantileValues'])],
                      # showline=True,
                      # showgrid=False,
                      # showticklabels=True,
                      # linecolor='rgb(204, 204, 204)',
                      # linewidth=2,
                      # ticks='outside',
                      # tickfont=dict(
                          # family='Arial',
                          # size=12,
                          # color='rgb(82, 82, 82)',
                      # ),
                  # ),
                   # yaxis=dict(
                      # showgrid=False,
                      # zeroline=False,
                      # showline=False,
                      # showticklabels=False,
                      # range = [0, max(res_data[:highest_term_index]["pdfValues"])]
                  # ),
                   # yaxis2=dict(
                      # showgrid=False,
                      # zeroline=False,
                      # showline=False,
                      # showticklabels=False,
                      # range = [0, max(res_data[:highest_term_index]["cumValue"])]
                  # ))
    # # total_graph.update_yaxes(range=[0,1])
    # frames = [dict(
                   # name = k,
                   # data = [go.Scatter(y= res_data.loc[res_data['term'] == f"{k} Terms",'pdfValues'], x=res_data.loc[res_data['term'] == f"{k} Terms",'quantileValues']),#update the trace 0
                               # go.Scatter(y = res_data.loc[res_data['term'] == f"{k} Terms",'cumValue'], x=res_data.loc[res_data['term'] == f"{k} Terms",'quantileValues'])#update the trace 2
                           # ],
                   # traces=[0,2]# the elements of the list [0,1,2] give info on the traces in fig.data
                                          # # that are updated by the above three go.Scatter instances
                  # ) for k in range(term+1) if (res_data['term'] == f"{k} Terms").any()]
    # # st.write(total_graph.layout)
    # updatemenus = [dict(type='buttons',
                                  # buttons=[dict(label='Play',
                                  # method='animate',
                                  # args=[[f'{k}' for k in range(term+1) if (res_data['term'] == f"{k} Terms").any()], 
                                         # dict(frame=dict(duration=750, redraw=False), 
                                              # transition=dict(duration=250),
                                              # easing='linear',
                                              # fromcurrent=True,
                                              # mode='immediate'
                                                                 # )])],
                    # direction= 'left', 
                    # pad=dict(r= 10, t=85), 
                    # showactive =True, x= 0.1, y= 0, xanchor= 'right', yanchor= 'top')
            # ]

    # sliders = [{'yanchor': 'top',
                # 'xanchor': 'left', 
                # 'currentvalue': {'font': {'size': 30}, 'prefix': 'Term: ', 'visible': True, 'xanchor': 'center'},
                # 'transition': {'duration': 500.0, 'easing': 'linear'},
                # 'pad': {'b': 10, 't': 50}, 
                # 'len': 0.9, 'x': 0.1, 'y': 0, 
                # 'steps': [{'args': [[k], {'frame': {'duration': 500.0, 'easing': 'linear', 'redraw': False},
                                          # 'transition': {'duration': 100, 'easing': 'linear'}}], 
                           # 'label': k, 'method': 'animate'} for k in range(term+1)  if (res_data['term'] == f"{k} Terms").any()     
                        # ]}]                       
    # total_graph.update(frames=frames)
    # total_graph.update_layout(updatemenus=updatemenus,
                  # sliders=sliders)
    # selected_points = plotly_events(total_graph)
    # st.write(selected_points)
    # if st.button("get data"):
    # total_graph
    # graphs_container.plotly_chart(total_graph, use_container_width=True)
    
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
    # alt_plot_all = None
    # for i in range(2,17):
        # # if (i-1)%6 == 0:
            # # alt_plot_all &= alt_plot_h
        # alt_plot = alt.Chart(InitialResults[str(i) + ' Terms'][['quantileValues','pdfValues']]).mark_line().encode(x='quantileValues',y='pdfValues').interactive()
        # # alt_plot2 = alt.Chart(InitialResults[str(i) + ' Terms'][['quantileValues','cumValue']]).mark_line().encode(x='quantileValues',y='cumValue').interactive() \
            # # + alt.Chart(m["dataValues"][['x','probs']]).mark_circle().encode(x='x',y='probs').interactive()
        # # if alt_plot_all is None:
            # # alt_plot_all = alt_plot | alt_plot2
        # # else:
            # # alt_plot_all &= alt_plot | alt_plot2
        # if alt_plot_all is None:
            # alt_plot_all = alt_plot
        # elif (i-2)%5 == 0:
            # alt_plot_all &= alt_plot
        # elif (i-2)%5 != 0:
            # alt_plot_all |= alt_plot
    # st.altair_chart(alt_plot_all, use_container_width=True)
    ################################################################
    
    
    
    ################################################################
    if big_plots:
        fig, ax = plt.subplots(1, 2, figsize=(10, 3), sharex='col')    
        # if st.session_state['mfitted'][key][name]['fit']
        # fig, ax = plt.subplots(1, 2, figsize=(4, 2), sharex='col')        
        # i = 2
        if term is None:
            pass
            # for i in range(results_len+1,1,-1):
                # if m['Validation']['valid'][i] == 'yes':
                    # j = 0
                    # # Plotting PDF
                    # ax[j].plot(InitialResults[str(i) + ' Terms']['quantileValues'], InitialResults[str(i) + ' Terms']['pdfValues'],
                          # linewidth=2, label=str(i) + ' Terms')
                    # # Plot data 
                    # ax[j + 1].scatter(m["dataValues"]['x'],m["dataValues"]['probs'],c='white',edgecolor='black', label=f'{name} Data')
                    # # Plotting CDF
                    # ax[j + 1].plot(InitialResults[str(i) + ' Terms']['quantileValues'], InitialResults[str(i) + ' Terms']['cumValue'],
                          # linewidth=2, label=str(i) + ' Terms')
                    # ax[j].patch.set_facecolor('white')
                    # # ax[j].axes.xaxis.set_ticks([])     
                    # # ax[j].axes.yaxis.set_ticks([])     
                    # ax[j + 1].patch.set_facecolor('white')
                    # ax[j + 1].axes.xaxis.set_ticks([])     
                    # # ax[j + 1].axes.yaxis.set_ticks([])     
                    # # ax[j].set(title=str(i) + ' Terms', ylabel='PDF', xlabel='Quantiles')
                    # # ax[j + 1].set(title=str(i) + ' Terms', ylabel='CDF', xlabel='Quantiles')
                    # ax[j].set(title=str(i) + ' Terms', ylabel='PDF')
                    # ax[j + 1].set(title=str(i) + ' Terms', ylabel='CDF')
                    # ax[j].axis([min(res_data['quantileValues']), max(res_data['quantileValues']), 0, max(res_data["pdfValues"])]) 
                    # ax[j+1].axis([min(res_data['quantileValues']), max(res_data['quantileValues']), round(min(m["dataValues"]['probs']),1), round(max(m["dataValues"]['probs']),1)]) 
                    # # if 'big plots' not in st.session_state['mfitted'][key][name]['plot']:
                        # # st.session_state['mfitted'][key][name]['plot']['big plot'] = plt 
                        # # return                        
                    # break
        else:
            # if st.session_state['mfitted'][key][name]['plot']['big plot']:
                # print("unpickled")
                # with open('sinus.pickle','rb') as fig_file:
                    # graphs_container.pyplot(pl.load(fig_file))
            # terms_for_loop = [term, max_valid_term]
            terms_for_loop = [term]
            # for i in range(2,term+1):
            for i in terms_for_loop :
                # print(m['Validation']['valid'])
                # print("i is",i)
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
                    # ax[j].legend(loc='upper center', bbox_to_anchor=(1.05, 0.05), fancybox=True, shadow=True, unsafe_allow_html=True)
                    # ax[j].legend(loc='upper center', unsafe_allow_html=True)
                    # ax[j + 1].legend([str(i) + ' Terms'])
                    # ax[j].axis([min(res_data['quantileValues']), max(res_data['quantileValues']), 0, max(res_data["pdfValues"])]) 
                    ax[j+1].axis([min(res_data['quantileValues']), max(res_data['quantileValues']), 0, 1]) 
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
            plt.tight_layout(rect=[0,0,0.75,1])
            graphs_container.pyplot(plt)       
            if st.session_state['mfitted'][key][name]['plot']['big plot'] is None:
                temp_img = io.BytesIO()
                plt.savefig(temp_img, format='png',transparent=True, unsafe_allow_html=True)
                st.session_state['mfitted'][key][name]['plot']['big plot'] = temp_img
                # graphs_container.pyplot(st.session_state['mfitted'][key][name]['plot']['big plot'])
                # img_from_bytes = Image.open(st.session_state['mfitted'][key][name]['plot']['big plot'])
                # graphs_container.image(st.session_state['mfitted'][key][name]['plot']['big plot'])
            # img_from_bytes = Image.open(st.session_state['mfitted'][key][name]['plot']['big plot'])
            # graphs_container.image(st.session_state['mfitted'][key][name]['plot']['big plot'],use_column_width = True)
                # graphs_container.pyplot(plt)
                # return
                    # break
        # ax[0].legend()
        # plt.tight_layout(rect=[0,0,0.75,1])
        # graphs_container.pyplot(plt)
        
        # plt.clf()
                       # ax[i-2, j].patch.set(title=str(current_term) + ' Terms', ylabel='PDF', xlabel='Quantiles')
                          
                              
                    # if current_term != 5*3:
                        # ax[i-2, j].set(title=str(current_term) + ' Terms', ylabel='CDF')
                    # else:
                       # ax[i-2, j].set(title=str(current_term) + ' Terms', ylabel='CDF', xlabel='Quantiles')fig, ax = plt.subplots(5, 3, figsize=(8, 3*3), sharex='col')
                       
        # i = 2
            ################################################################
    if csv:
        if st.session_state['mfitted'][key][name]['plot']['csv'] is None:
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
                          
            plt.tight_layout(rect=[0,0,0.75,1])
            temp_img = io.BytesIO()
            plt.savefig(temp_img, format='png',transparent=True, unsafe_allow_html=True)
            st.session_state['mfitted'][key][name]['plot']['csv'] = temp_img
            # graphs_container.pyplot(st.session_state['mfitted'][key][name]['plot']['big plot'])
            # img_from_bytes = Image.open(st.session_state['mfitted'][key][name]['plot']['big plot'])
            # graphs_container.image(st.session_state['mfitted'][key][name]['plot']['big plot'])
            # img_from_bytes = Image.open(st.session_state['mfitted'][key][name]['plot']['big plot'])
            graphs_container_main.image(st.session_state['mfitted'][key][name]['plot']['csv'],use_column_width = True)
            print("looped")
        else:
            graphs_container_main.image(st.session_state['mfitted'][key][name]['plot']['csv'],use_column_width = True)
    
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
                                       name,
                                       user_term):
    #Create metalog
    # st.write(boundedness,
                       # bounds)
    if 'mfitted' not in st.session_state:
      st.session_state['mfitted'] = {'csv':{},'quantile':{}}
    if probs is np.nan:
      key = 'csv'
    else:
      key = 'quantile'
    # update_boundedness(False)
    if (name not in st.session_state['mfitted'][key] or st.session_state['mfitted'][key][name]['fit'] is None) or (name in st.session_state['mfitted'][key] and not user_term is None and st.session_state['mfitted'][key][name]['fit']['Validation']['term'].max() < user_term ):
      print(f"running metalog fit for {name} in {key}")
      mfitted = metalog.fit(x, bounds = bounds, boundedness = boundedness, fit_method ='OLS', term_limit = terms, probs=probs)
      # max_valid_term = int(mfitted['Validation'][(mfitted['Validation']['valid'] == 'yes') & (mfitted['Validation']['term'] <= user_term)]['term'].max())
      st.session_state['mfitted'][key][name] = {'fit':mfitted,'plot':{'csv':None,'big plot':None},'options':{'boundedness':boundedness, 'terms':user_term, 'bounds': bounds}}
    print("user term is",user_term)
    #Create graphs
    # st.write(st.session_state['mfitted'][key][name]['fit'].keys())
    # st.write(type(st.session_state['mfitted'][key][name]['fit']))
    # st.write(st.session_state['mfitted'][key][name]['fit']['M'])
    # st.write(st.session_state['mfitted'][key][name]['fit']['A'])
    # st.write(st.session_state['mfitted'][key][name]['fit']["dataValues"])
    # st.write(st.session_state['mfitted'][key][name]['fit']['Validation'])
    # big_plots = st.sidebar.checkbox("Big Graphs?")
    # max_valid_term = int(st.session_state['mfitted'][key][name]['fit']['Validation'][st.session_state['mfitted'][key][name]['fit']['Validation']['valid'] == 'yes']['term'].max())
    # print(type(max_valid_term))
    # if big_plots:
        # term = graphs_container.slider(f"Select The Maximum Number of Terms for the {name} Current Variable: ",2,max_valid_term,value=5,key=f"{name} term slider")
    # else:
        # term = int(16) 
    # plot(mfitted, True,csv=None,term=None,name=name)
    plot(st.session_state['mfitted'][key][name]['fit'],big_plots,csv,user_term,name=name,key=key)

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
 
def convert_to_number(value):
    
    if isinstance(value,dict):
        value = {k:float(v) if isinstance(v,str) and (v.isnumeric() or (data_type_str == 'csv' and v != 'PM_Index')) else v for k,v in value.items()}
    return value

def update_max_term(variable_index,variable_name):
    if 'quantiles_data' in st.session_state and variable_name in st.session_state['quantiles_data'] and 'number_of_quantiles' in st.session_state['quantiles_data'][variable_name]:
        st.session_state['quantiles_data'][variable_name]['number_of_quantiles'] = st.session_state[f"Quantile{variable_index}"]
        st.session_state['quantiles_data'][variable_name].pop('col_terms',None)
        st.session_state['quantiles_data'][variable_name].pop('q_data',None)
        
def update_terms(selected_column, data_type='csv', variable_index=0):
    if data_type == 'csv':
        value = st.session_state["Column_Terms"]
        if 'mfitted' in st.session_state:
            if selected_column not in st.session_state['mfitted'][data_type]:
                print("selected_column",selected_column)
                # st.session_state['mfitted'][data_type][selected_column]['options']['terms'] = value
            elif st.session_state['mfitted'][data_type][selected_column]['options']['terms'] != value:
                st.session_state['mfitted'][data_type][selected_column]['options']['terms'] = value
    else:
        if 'quantiles_data' in st.session_state and selected_column in st.session_state['quantiles_data'] and 'col_terms' in st.session_state['quantiles_data'][selected_column]:
            st.session_state['quantiles_data'][selected_column]['col_terms'] = st.session_state[f"Column_Terms {selected_column} {variable_index}"]
def update_values(variable_name, session_key, row):
    if 'quantiles_data' in st.session_state and variable_name in st.session_state['quantiles_data'] and 'q_data' in st.session_state['quantiles_data'][variable_name]:
        # st.session_state['quantiles_data'][quantile_name[quantile_index]]['q_data']
        df = st.session_state['quantiles_data'][variable_name]['q_data'].reset_index()
        col = 0 if session_key[0] == 'y' else 1
        print("value in df is",df.iloc[row,col])
        df.iloc[row,col] = st.session_state[session_key]
        print("value in df is",df.iloc[row,col])
        print("df is",df.set_index(""))
        st.session_state['quantiles_data'][variable_name]['q_data'] = df.set_index("")
        if 'mfitted' in st.session_state and variable_name in st.session_state['mfitted'][data_type_str]:
          st.session_state['mfitted'][data_type_str].pop(variable_name,None)
def update_correlations(session_key):
    if 'quantiles_data' in st.session_state and 'correlations' in st.session_state['quantiles_data']:
        # st.session_state['quantiles_data'][quantile_name[quantile_index]]['q_data']
        df = st.session_state['quantiles_data']['correlations']
        col,row = session_key.split(" vs ")
        print("col,row",col,row)
        print("value in df is",df.loc[row,col])
        df.loc[row,col] = st.session_state[session_key]
        print("value in df is",df.loc[row,col])
        print("df is",df)
        st.session_state['quantiles_data']['correlations'] = df      

def update_input_name():
  print("Big Graph Column is", st.session_state["Big Graph Column"])
  st.session_state["quantile_counter"]  = st.session_state['quantiles_data'][st.session_state["Big Graph Column"]]['pos']

def update_variable_count():
  if "Number of Quantiles Variables" in st.session_state:
    st.session_state['quantiles_variable_count'] = st.session_state["Number of Quantiles Variables"] 

def update_counter(value):
  if data_type_str == 'quantile':
    if 'quantile_counter' in st.session_state and 'Number of Quantiles Variables' in st.session_state:
      if (value == -1 and st.session_state['quantile_counter'] == 1) or (value == 1 and st.session_state['quantile_counter'] == st.session_state['Number of Quantiles Variables']):
        return None
      st.session_state['quantile_counter'] += value
    else:
      st.session_state['quantile_counter'] = 1
  else:
    if 'csv_counter' in st.session_state and 'column_index' in st.session_state:
      if (value == -1 and st.session_state['csv_counter'] == 1) or (value == 1 and st.session_state['csv_counter'] == len(st.session_state["column_index"])):
        return None
      st.session_state['csv_counter'] += value
      print("keys",list(st.session_state["column_index"].keys())[st.session_state['csv_counter'] - 1])
      st.session_state["Big Graph Column"] = list(st.session_state["column_index"].keys())[st.session_state['csv_counter'] - 1]
    else:
      st.session_state['csv_counter'] = 1
    
    
def update_name(num):
  new_name = st.session_state[f"Quantile Name {num}"]
  if not new_name in st.session_state['quantiles_data']:
    for variable in st.session_state['quantiles_data']:
      if st.session_state['quantiles_data'][variable]['pos'] == num:
        st.session_state['quantiles_data'][new_name] = st.session_state['quantiles_data'].pop(variable)
        if 'mfitted' in st.session_state and variable in st.session_state['mfitted'][data_type_str]:
          st.session_state['mfitted'][data_type_str][new_name] = st.session_state['mfitted'][data_type_str].pop(variable,None)
        break
  pass
def update_boundedness(refresh = False, data_type = 'csv',max = 1,min = 0, quantile_count = None, variable_name=None):
    if quantile_count is None:
        boundedness = st.session_state["Column_boundedness"]
    else:
        boundedness = st.session_state[f"Column_boundedness {variable_name} {quantile_count+1}"]
    
    selected_column = st.session_state["Big Graph Column"] if data_type == 'csv' else variable_name
    print("boundedness from session is ",boundedness)
    print("selected_column from session is ",selected_column)
    if "Column_upper" not in st.session_state:
        print("Column_upper in session")
        upper = max
    else:
        upper = st.session_state["Column_upper"]
    if "Column_lower" not in st.session_state:
        print("Column_lower in session")
        lower = min
    else:
        lower = st.session_state["Column_lower"]
       
    #convert to float and list 
    if boundedness == "'b' - bounded on both sides":
        bounds = [lower, upper]
    elif boundedness.find("lower") != -1:
        bounds = [lower]
    elif boundedness.find("upper") != -1:
        bounds = [upper]
    else:
        bounds = [0,1] 
    boundedness = boundedness.strip().split(" - ")[0].replace("'","")     
    
    if 'quantiles_data' in st.session_state and selected_column in st.session_state['quantiles_data'] and 'boundedness' in st.session_state['quantiles_data'][selected_column]:
        st.session_state['quantiles_data'][selected_column]['boundedness'] = boundedness
        st.session_state['quantiles_data'][selected_column]['bounds'] = bounds

    if 'mfitted' in st.session_state and selected_column in st.session_state['mfitted'][data_type]:
        if selected_column not in st.session_state['mfitted'][data_type] or (st.session_state['mfitted'][data_type][selected_column]['options']['boundedness'] != boundedness or st.session_state['mfitted'][data_type][selected_column]['options']['bounds'] != bounds):
            # print("selected_column",selected_column)
            if any([x[0] != x[1] for x in zip(st.session_state['mfitted'][data_type][selected_column]['options']['boundedness'] , boundedness)]):
                print("saved",st.session_state['mfitted'][data_type][selected_column]['options']['boundedness'] , "current boundedness",boundedness)
                st.session_state['mfitted'][data_type][selected_column]['options']['boundedness'] = boundedness
                refresh = True
            if any([float(x[0]) != float(x[1]) for x in zip(st.session_state['mfitted'][data_type][selected_column]['options']['bounds'], bounds)]):
                print("saved",st.session_state['mfitted'][data_type][selected_column]['options']['bounds'] , "current bounds",bounds)
                st.session_state['mfitted'][data_type][selected_column]['options']['bounds'] = bounds
                print("saved after saving",st.session_state['mfitted'][data_type][selected_column]['options']['bounds'] , "current bounds",bounds)
                refresh = True
        # elif :
            # st.session_state['mfitted'][data_type][selected_column]['options']['boundedness'] = boundedness
            # st.session_state['mfitted'][data_type][selected_column]['options']['boundedness'] = boundedness
            # st.session_state['mfitted'][data_type][selected_column]['options']['bounds'] = bounds
            # st.session_state['mfitted'][data_type][selected_column]['options']['bounds'] = bounds
            #TODO: recalculate when bounds change
    if refresh and 'mfitted' in st.session_state and selected_column in st.session_state['mfitted'][data_type]:
        st.session_state['mfitted'][data_type][selected_column]['fit'] = None
        # st.session_state['mfitted']['quantile'][selected_column]['fit'] = None
        st.session_state['mfitted'][data_type][selected_column]['plot'] = {data_type:None,'big plot':None}
        # st.session_state['mfitted']['quantile'][selected_column]['plot'] = {data_type:None,'big plot':None}
def update_seeds(data_type='csv',  entity = None,  varId = None,  seed3 = None,  seed4 = None, variable_name=None):                 
    selected_column = st.session_state["Big Graph Column"] if data_type == 'csv' else variable_name
    
    if data_type != 'csv' and 'quantiles_data' in st.session_state and selected_column in st.session_state['quantiles_data'] and 'seeds' in st.session_state['quantiles_data'][selected_column]:
        for item in ['entity','seed3','seed4']:
          if st.session_state[f"{item} {selected_column}"].isnumeric():
            st.session_state['quantiles_data'][selected_column]['seeds']['arguments'][item] = st.session_state[f"{item} {selected_column}"] 
          else:
            st.warning(f'{item} must be a number.')
            # st.stop()
        if st.session_state[f"varId {selected_column}"].strip():
          st.session_state['quantiles_data'][selected_column]['seeds']['arguments']['varId'] = st.session_state[f"varId {selected_column}"] 
        else:
            st.warning(f'varId must have a value.')
            # st.stop()
          
        st.session_state['quantiles_data'][selected_column]['seeds']['arguments'] = {'counter':'PM_Index',
                                                                                'entity' : st.session_state[f"entity {selected_column}"] ,
                                                                                'varId' : st.session_state[f"varId {selected_column}"] ,
                                                                                'seed3' : st.session_state[f"seed3 {selected_column}"],
                                                                                'seed4' : st.session_state[f"seed4 {selected_column}"] }    
    
    if 'mfitted' in st.session_state and selected_column in st.session_state['mfitted'][data_type] and 'seeds' in st.session_state['mfitted'][data_type][selected_column]['options']:
        if not entity is None:
            st.session_state['mfitted'][data_type][selected_column]['options']['seeds']['arguments']['entity'] = entity   
        elif not varId is None:
            st.session_state['mfitted'][data_type][selected_column]['options']['seeds']['arguments']['varId'] = varId   
        elif not seed3 is None:
            st.session_state['mfitted'][data_type][selected_column]['options']['seeds']['arguments']['seed3'] = seed3   
        elif not seed4 is None:
            st.session_state['mfitted'][data_type][selected_column]['options']['seeds']['arguments']['seed4'] = seed4   
        elif selected_column in st.session_state['mfitted'][data_type]:
            st.session_state['mfitted'][data_type][selected_column]['options']['seeds']['arguments'] = {'counter':'PM_Index',
                                                                                    'entity' : st.session_state[f"entity {selected_column}"] ,
                                                                                    'varId' : st.session_state[f"varId {selected_column}"] ,
                                                                                    'seed3' : st.session_state[f"seed3 {selected_column}"],
                                                                                    'seed4' : st.session_state[f"seed4 {selected_column}"] }    
def make_csv_graph(series,
                                       probs,
                                       boundedness,
                                       bounds,
                                       big_plots,
                                       user_terms,
                                       graphs):
    if big_plots:
        graphs_container.markdown(f"<div id='linkto_head'></div>", unsafe_allow_html=True)
        graphs_container.header(series.name)
        print(probs)
    preprocess_charts(series.to_list(),
                                    probs,
                                    boundedness,
                                    bounds,
                                    big_plots,
                                    16 if probs is np.nan else user_terms,
                                    graphs,
                                    series.name,
                                    user_terms)
    # if probs is np.nan:
        # preprocess_charts(series.to_list(),
                                        # probs,
                                        # boundedness,
                                        # bounds,
                                        # big_plots,
                                        # 16,
                                        # graphs,
                                        # series.name,
                                        # user_terms)
    # else:
        # preprocess_charts(series.to_list(),
                                        # probs,
                                        # boundedness,
                                        # bounds,
                                        # big_plots,
                                        # user_terms,
                                        # graphs,
                                        # series.name,
                                        # user_terms)
                                    
    return None
# @st.cache
def input_data(name,i,df,probs=None):
    if probs is None:
        probs = np.nan
        max_val = 16
        default_val = 5
        input_data_type = 'csv'
        data_columns = df.columns
        table_container.write("If the data above appears correct, enter your parameters in the sidebar for this file.")
    else:
        max_val = df.shape[0]
        default_val = max_val
        input_data_type = 'quantile'
        data_columns = df.columns
    
    with st.sidebar.expander("Output Options"):
        filename_container = st.container()
        file_name_no_ext, file_ext = filename_container.columns(2)
        filename = file_name_no_ext.text_input(f'Filename {i+1}', name,key=f"{name}_{i}_filename") + '.SIPmath'
        file_ext.write('File Extension')
        file_ext.write('.SIPmath')
        author = st.text_input(f'Author for {filename}', 'Unknown',key=f"{name}_author")
        # if data_type_str != "quantile":
        dependent_name = 'Guassian Copula' if data_type_str == "csv" else "Correlation Matrix"
        dependence = st.selectbox('Dependence', ('independent',dependent_name),key=f"{name}_{i}_dependence")
        correlation_df = None
        if dependence == "Correlation Matrix":
          number_variables = len(data_columns)
          print(number_variables)
          correlation_container = graphs_container.columns(number_variables)
          if 'quantiles_data' in st.session_state and 'correlations' in st.session_state['quantiles_data']:
            saved_df = st.session_state['quantiles_data']['correlations']
            for row in df.columns:
              for col in df.columns:
                print("value in saved_df is",saved_df.loc[row,col])
                st.session_state[f"{row} vs {col}"] = saved_df.loc[col,row]
          corrs_data = [[float(correlation_container[i-1].number_input(f'{df.columns[j]} vs {df.columns[i-1]} ', 
               value = 1.0 if j+1 == i else 0.0,
               format = "%f",
               # step = 0.1
                on_change = update_correlations,
                args = (f"{df.columns[j]} vs {df.columns[i-1]}",),
                key=f"{df.columns[j]} vs {df.columns[i-1]}"))  for j in range(i)] for i in range(1,number_variables+1)]
          print("corrs_data", corrs_data)
          correlation_df = pd.DataFrame(corrs_data,columns=df.columns,index=df.columns)
          st.session_state['quantiles_data']['correlations'] = correlation_df
          check_correlation_df = correlation_df.fillna(correlation_df.T)
          print("check_correlation_df",check_correlation_df)
          correlation_check_run = True
          if graphs_container.button('Store Correlation', key='correlation_done'):
            print('ran correlation check')
            print(np.linalg.eigvals(check_correlation_df.to_numpy()))
            correlation_check_run = np.all(np.linalg.eigvals(check_correlation_df.to_numpy()) > 0)
            if not correlation_check_run:
              graphs_container.warning("The correlation matrix needs to be feasible.")
              # st.stop()
            
          print(correlation_df)
            # pass
        if dependence != 'independent':
            dependence = 'dependent'
        # else:
            # dependence = 'independent'
        # boundedness = st.selectbox('Boundedness', ("'u' - unbounded", 
                                                           # "'sl' - semi-bounded lower", 
                                                           # "'su' - semi-bounded upper",
                                                           # "'b' - bounded on both sides"),key=f"{name}_boundedness")
                                                           
        # if boundedness == "'b' - bounded on both sides":
            # #convert to float and list
            # boundsl = st.text_input('Lower Bound', '0',key=f"{name}_lower")
            # boundsu = st.text_input('Upper Bound', '1',key=f"{name}_upper")
            # bounds = [float(boundsl),float(boundsu)]
        # elif boundedness.find("lower") != -1:
            # bounds = [float(st.text_input('Lower Bound', '0',key=f"{name}_lower"))]
        # elif boundedness.find("upper") != -1:
            # bounds = [float(st.text_input('Upper Bound', '1',key=f"{name}_upper"))]
        # else:
            # bounds = [0,1]
            
        # boundedness = boundedness.strip().split(" - ")[0].replace("'","")
        # if max_val > 3:
            # term_saved = st.slider('Term Saved',3,max_val,default_val,key=f"{name}_term_saved")
        # else:
             # term_saved = 3
        if 'mfitted' in st.session_state:
            fit_data = [not st.session_state['mfitted'][input_data_type][x]['fit'] is None
                                for x in data_columns 
                                    if x in st.session_state['mfitted'][input_data_type]]     
            # for x in data_columns:
                # if x in st.session_state['mfitted'][input_data_type]:
                    # st.session_state['mfitted'][input_data_type][x]['options']['seeds'] = st.session_state['quantiles_data'][x]['seeds']
                                                     
            seeds_data = ['seeds' in st.session_state['mfitted'][input_data_type][x]['options'] 
                                        for x in data_columns 
                                            if x in st.session_state['mfitted'][input_data_type]] 
        # print("fit_data",fit_data)
        # print("seeds_data",seeds_data)
         # ###### I need to fix the problem by adjusting csv to variable then select it based on if probs is None or not.
        if 'mfitted' in st.session_state and all(fit_data) and all(seeds_data):
            term_saved = [st.session_state['mfitted'][input_data_type][x]['options']['terms'] if x in st.session_state['mfitted'][input_data_type] else None for x in data_columns]
            print("term_saved is ",type(term_saved[0]))
            if all(term_saved) and "--------Enter number of terms--------" not in term_saved:
                print(f'Writing SIPmath with data_columns as {data_columns}')
                boundedness = [st.session_state['mfitted'][input_data_type][x]['options']['boundedness'] if x in st.session_state['mfitted'][input_data_type] else None for x in data_columns]
                bounds = [[y for y in st.session_state['mfitted'][input_data_type][x]['options']['bounds']] if x in st.session_state['mfitted'][input_data_type] else None for x in data_columns]
                print(bounds)
                preview_options = pd.DataFrame(term_saved, index = data_columns, columns = ['Term'])
                preview_options['Boundedness'] = boundedness
                preview_options['Lower Bounds'] = [float(x[0]) if y == 'b'  or y == 'sl' else np.nan for x,y in zip(bounds,boundedness) ]
                preview_options['Upper Bounds'] = [float(x[-1]) if y == 'b'  or y == 'su' else np.nan for x,y in zip(bounds,boundedness) ]
                print(preview_options['Term'])
                for coeff_num in range(int(preview_options['Term'].max())):
                    # preview_options[f'A {coeff_num + 1}'] = [st.session_state['mfitted'][input_data_type][x]['fit']['A'].iloc[coeff_num,st.session_state['mfitted'][input_data_type][x]['options']['terms'] - 1] for x in data_columns]
                    preview_options[f'A {coeff_num + 1}'] = [st.session_state['mfitted'][input_data_type][x]['fit']['A'].iloc[coeff_num,st.session_state['mfitted'][input_data_type][x]['options']['terms'] - 1] 
                                                                                if st.session_state['mfitted'][input_data_type][x]['fit']['A'].shape[0] > coeff_num else None for x in data_columns]
                # preview_options = preview_options.T
                table_container.write(preview_options)
                converted_seeds = [{k:convert_to_number(v) for k,v in st.session_state['mfitted'][input_data_type][x]['options']['seeds'].items()} for x in data_columns ]
                print("converted_seeds is",converted_seeds)
                if st.button(f'Convert to {filename.split(".")[0]} SIPmath Json?',key=f"{filename.split('.')[0]}_term_saved"):
                    table_container.subheader("Preview and Download the JSON file below.")
                    data_dict_for_JSON = dict(boundedness=boundedness,
                                   bounds=bounds,
                                   term_saved=term_saved)
                    print("probs in JSON", probs)
                    PySIP.Json(SIPdata=df,
                                   file_name=filename,
                                   author=author,
                                   dependence=dependence,
                                   setupInputs=data_dict_for_JSON,
                                   seeds = converted_seeds,
                                   probs=probs,
                                   quantile_corr_matrix = correlation_df)
                                   
                    with open(filename) as f:
                        st.download_button(
                                label=f"Download {filename}",
                                data=f,
                                file_name=filename
                                )
                # st.text("Copy the link below to paste into SIPmath.")
                # with open(filename, 'rb') as f:
                    # st.write(sent_to_pastebin(filename,f.read()).text.replace("https://pastebin.com/","https://pastebin.com/raw/"))
                    table_container.text("Mouse over the text then click on the clipboard icon to copy to your clipboard.")
                    with open(filename, 'rb') as f:
                        table_container.json(json.load(f))
        else:
            st.warning(f'{filename}.SIPmath cannot be saved until all variables have been configured.')

#st.title('SIPmath JSON Creator')
st.sidebar.header('User Input Parameters')

# Collects user input features into dataframe
data_type = st.sidebar.radio('Input Data Type', ('CSV File','Quantile'), index=0)
data_type_str = data_type.split()[0].lower()
if data_type == 'CSV File':
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"],accept_multiple_files=False)
    if uploaded_file != None:
        input_df = None
        uploaded_file = uploaded_file if isinstance(uploaded_file, list) else [uploaded_file]
        for i,file in enumerate(uploaded_file):
            try:
                input_df = pd.read_csv(file)
            except UnicodeDecodeError:
                input_df = pd.read_csv(file,encoding='cp437')
            # [for x in i]
            name = file.name.replace(".csv","")
            # with main_contanier.container():
            table_container.subheader(f"Preview for {name}")
            table_container.write(input_df[:10].to_html(index=False), unsafe_allow_html=True)
        if isinstance(input_df, pd.DataFrame):
            # if "column_index" not in st.session_state:
            st.session_state["column_index"] = {k:v for v,k in enumerate(input_df.columns)}
            print("column_index = ",st.session_state["column_index"])
            # if "selected_column" in locals():
                # print('selected column is here!')
                # print("selected_column",selected_column)
                # if 'mfitted' in st.session_state  and selected_column in st.session_state['mfitted'][data_type_str]:
                    # print("selected_column",selected_column)
                    # st.session_state["Column_Terms"] = "-"
                
            # boundedness = boundedness.strip().split(" - ")[0].replace("'","")
            # terms_list = ["-",*list(range(2,17))]
            # col_terms = quanile_container.selectbox(f'Current Variable: Number of Terms', terms_list, index=0, key=f"Column_Terms")
            # print(st.session_state["Column_Terms"])
            # col_terms = int(col_terms) if isinstance(col_terms,int) else None                        
            # big_plots = quanile_container.checkbox("Big Graphs?",value=False)
            seed_container = st.sidebar.container()
    # big_plots, make_graphs_checkbox = True, True
            selected_column = graphs_container.selectbox("Select Current Variable:",  input_df.columns,key="Big Graph Column")     
            #on change pass the max, min value for the column
            boundedness = seed_container.selectbox(f'Current Variable: Boundedness', ("'u' - unbounded", 
                                                       "'sl' - semi-bounded lower", 
                                                       "'su' - semi-bounded upper",
                                                       "'b' - bounded on both sides"),
                                                       # on_change = update_boundedness,
                                                       key=f"Column_boundedness")
                                                                                                
            if boundedness == "'b' - bounded on both sides":
                #convert to float and list
                boundsl = seed_container.number_input('Lower Bound',max_value  =  input_df[selected_column].min(),value  =  input_df[selected_column].min(),format = "%f",key=f"Column_lower") 
                # if 
                # seed_container.number_input('Lower Bound')
                boundsu = seed_container.number_input('Upper Bound',max_value  = input_df[selected_column].max(),value  = input_df[selected_column].max(),key=f"Column_upper")

                # seed_container.number_input('Upper Bound')
                bounds = [float(boundsl),float(boundsu)]
                if 'mfitted' in st.session_state and selected_column in st.session_state['mfitted'][data_type_str]:
                    update_boundedness()
            elif boundedness.find("lower") != -1:
                boundsl = seed_container.number_input('Lower Bound',max_value  =  input_df[selected_column].min(),value  =  input_df[selected_column].min(),format = "%f",key=f"Column_lower") 
                # if not str(boundsl).isnumeric()  or (str(boundsl).isnumeric()  and float(boundsl) > float(input_df[selected_column].min())):
                    # # lower_bound.error(f"The number needs to be equal to or less than the minimum value ({input_df[selected_column].min()}) of {selected_column}")
                    # boundsl = input_df[selected_column].min()
                bounds = [float(boundsl)]
                if 'mfitted' in st.session_state and selected_column in st.session_state['mfitted'][data_type_str]:
                    print(f'The bound is {bounds} before saving.')
                    update_boundedness(min = input_df[selected_column].min())
            elif boundedness.find("upper") != -1:
                boundsu = seed_container.number_input('Upper Bound',max_value  = input_df[selected_column].max(),value  = input_df[selected_column].max(),key=f"Column_upper")
                bounds = [float(boundsu)]
                if 'mfitted' in st.session_state and selected_column in st.session_state['mfitted'][data_type_str]:
                    update_boundedness()
            else:
                bounds = [0,1]      
                if 'mfitted' in st.session_state and selected_column in st.session_state['mfitted'][data_type_str]:
                    update_boundedness()
            
            if "selected_column" in locals():
                print('selected column is here!')
                print("selected_column",selected_column)
                if 'mfitted' in st.session_state and selected_column in st.session_state['mfitted'][data_type_str]:
                    print("selected_column",selected_column)
                    st.session_state["Column_Terms"] = st.session_state['mfitted'][data_type_str][selected_column]['options']['terms'] if st.session_state['mfitted'][data_type_str][selected_column]['options']['terms'] else "--------Enter number of terms--------"
                
            boundedness = boundedness.strip().split(" - ")[0].replace("'","")
            terms_list = ["--------Enter number of terms--------",*list(range(3,17))]
            # col_terms = None
            
            col_terms = seed_container.selectbox(f'Current Variable: Number of Terms', 
                                terms_list, 
                                on_change=update_terms, 
                                args=(selected_column,), 
                                key=f"Column_Terms")
                #add SPT normal or three terms
            default_column = input_df.columns[0]
            print(st.session_state["Column_Terms"])
            col_terms = int(col_terms) if isinstance(col_terms,int) else 3 
            left_buttons, right_buttons = seed_container.columns(2)
            seed_container.write("Enter HDR Seed Values Below:")
            left_values, right_values = seed_container.columns(2)
            make_graphs_checkbox = left_buttons.button("Make Graph Panel")   
            if not make_graphs_checkbox:
              input_df[[selected_column]].apply(make_csv_graph,
                                             probs = np.nan,
                                             boundedness = boundedness,
                                             bounds = bounds,
                                             big_plots = True,
                                             user_terms = col_terms,
                                             graphs = False )
            if right_buttons.button("Apply to All Variables"):
              for i,col in enumerate(input_df.columns):
                if not col in  st.session_state['mfitted'][data_type_str]:
                  input_df[[col]].apply(make_csv_graph,
                                                       probs = np.nan,
                                                       boundedness = boundedness,
                                                       bounds = bounds,
                                                       big_plots = True,
                                                       user_terms = col_terms,
                                                       graphs = False )
                  st.session_state['mfitted'][data_type_str][col]['options']['seeds'] = {
                                                                             'name':'hdr'+str(st.session_state["column_index"][selected_column]+1),
                                                                             'function':'HDR_2_0',
                                                                             'arguments': {'counter':'PM_Index',
                                                                                                  'entity' : "0" ,
                                                                                                  'varId' : str(i+1),
                                                                                                  'seed3' : "0",
                                                                                                  'seed4' : "0" }}
                  if default_column != col and st.session_state['mfitted'][data_type_str][col]['options']['seeds']['arguments']["varId"] is None:
                      st.session_state['mfitted'][data_type_str][col]['options']['seeds']['arguments']["varId"]  =  str(float(st.session_state['mfitted'][data_type_str][col]['options']['seeds']['arguments']["varId"] ) + st.session_state["column_index"][col])
             
                print('assigned seeds')
            
            previous_button =  left_buttons.button("Previous Variable",on_click = update_counter,args=(-1,),key=f"previous_button")
            next_button = right_buttons.button("Next Variable",on_click = update_counter,args=(1,),key=f"next_button") 
            if 'mfitted' in st.session_state and selected_column in st.session_state['mfitted'][data_type_str] and 'seeds' not in st.session_state['mfitted'][data_type_str][selected_column]['options']:
                    st.session_state['mfitted'][data_type_str][selected_column]['options']['seeds'] = {
                                                                                               'name':'hdr'+str(st.session_state["column_index"][selected_column]+1),
                                                                                               'function':'HDR_2_0',
                                                                                               'arguments': {'counter':'PM_Index',
                                                                                                                    'entity' : "0" ,
                                                                                                                    'varId' : None,
                                                                                                                    'seed3' : "0",
                                                                                                                    'seed4' : "0" }}
            elif 'mfitted' in st.session_state and f"entity {selected_column}" in st.session_state and st.session_state[f"entity {selected_column}"] == "":
                st.session_state['mfitted'][data_type_str][selected_column]['options']['seeds']['arguments'] = {'entity' : st.session_state[f"entity {selected_column}"] ,
                                                                                            'varId' : st.session_state[f"varId {selected_column}"] ,
                                                                                            'seed3' : st.session_state[f"seed3 {selected_column}"],
                                                                                            'seed4' : st.session_state[f"seed4 {selected_column}"] }
            
            if 'mfitted' in st.session_state and st.session_state['mfitted'][data_type_str][default_column]['options']['seeds']['arguments']["varId"] is None:
                st.session_state['mfitted'][data_type_str][default_column]['options']['seeds']['arguments']["varId"] = "1"
           
            if default_column != selected_column and st.session_state['mfitted'][data_type_str][selected_column]['options']['seeds']['arguments']["varId"] is None:
                st.session_state['mfitted'][data_type_str][selected_column]['options']['seeds']['arguments']["varId"]  =  str(float(st.session_state['mfitted'][data_type_str][default_column]['options']['seeds']['arguments']["varId"] ) + st.session_state["column_index"][selected_column])
             
            if 'mfitted' in st.session_state and selected_column in st.session_state['mfitted'][data_type_str]:
              st.session_state[f"entity {selected_column}"] = str(st.session_state['mfitted'][data_type_str][selected_column]['options']['seeds']['arguments']["entity"])
              st.session_state[f"varId {selected_column}"] = str(st.session_state['mfitted'][data_type_str][selected_column]['options']['seeds']['arguments']["varId"])
              st.session_state[f"seed3 {selected_column}"] = str(st.session_state['mfitted'][data_type_str][selected_column]['options']['seeds']['arguments']["seed3"])
              st.session_state[f"seed4 {selected_column}"]  = str(st.session_state['mfitted'][data_type_str][selected_column]['options']['seeds']['arguments']["seed4"])  
            seed_data = [left_values.text_input(f'entity', value="0", key=f"entity {selected_column}"),
                            right_values.text_input(f'varId', value="1", key=f"varId {selected_column}"),
                            left_values.text_input(f'seed 3',value="0",  key=f"seed3 {selected_column}"),
                            right_values.text_input(f'seed 4',value="0",  key=f"seed4 {selected_column}")]                                                                   
            update_seeds(data_type_str)
            print(f"seed_data is {seed_data}")
            big_plots = True
            if make_graphs_checkbox or all(input_df.any()):
                if make_graphs_checkbox:
                  # pass
                    # if not "selected_column" in locals():
                        # empty_table.empty()
                    # update_boundedness()
                    # upper_bound, lower_bound = seed_container.columns(2)
                    #TODO: error handling for value too small for upper bounds or letter    
                    input_df[[selected_column]].apply(make_csv_graph,
                                                   probs = np.nan,
                                                   boundedness = boundedness,
                                                   bounds = bounds,
                                                   big_plots = big_plots,
                                                   user_terms = col_terms,
                                                   graphs = make_graphs_checkbox )
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
    quanile_container = st.sidebar.container()
    # quantile_number_variable = 1
    if 'quantiles_variable_count' in st.session_state:
      # st.session_state["Number of Quantiles Variables"] = st.session_state['quantiles_variable_count']
      st.session_state["Number of Quantiles Variables"] = st.session_state['quantiles_variable_count']
    quantile_number_variable = int(quanile_container.number_input('Number of Variables: ',
                                    value=int(1),
                                    min_value = 1,
                                    on_change = update_variable_count,
                                    key="Number of Quantiles Variables"))
    st.session_state['quantiles_variable_count'] = quantile_number_variable
                                    # I could write the data into a dict or dataframe then look it up using variable name and variable type
    if not 'quantiles_data' in st.session_state:
        st.session_state['quantiles_data'] = {}
        # quantile_name = [] 
        # q_data = [] 
        # number_of_quantiles = [] 
        # terms_list = [] 
        # col_terms = [] 
        # boundedness= []
    else:
        # st.session_state['quantiles_data']
        # quantile_name = [x for x in st.session_state['quantiles_data']] 
        # q_data = [st.session_state['quantiles_data'][x]['q_data'] for x in quantile_name] 
        # number_of_quantiles = [st.session_state['quantiles_data'][x]['number_of_quantiles'] for x in quantile_name] 
        # terms_list = [st.session_state['quantiles_data'][x]['terms_list'] for x in quantile_name] 
        # col_terms = [st.session_state['quantiles_data'][x]['col_terms'] for x in quantile_name] 
        # boundedness = [st.session_state['quantiles_data'][x]['boundedness'] for x in quantile_name] 
        pass
    with quanile_container.expander("Variable Input", expanded=True):      
        # for num in range(1,quantile_number_variable+1): 
        if not 'quantile_counter' in st.session_state:
          st.session_state['quantile_counter'] = 1
        num = st.session_state['quantile_counter']
        quantile_index = num - 1
        st.header(f"Variable {num}'s Inputs")
        current_variable_name = f'x{num}'
        try:
          for x in st.session_state['quantiles_data']:
            if st.session_state['quantiles_data'][x]['pos'] == num:
              current_variable_name = x
              break
        except KeyError:
          pass 
        if not current_variable_name in st.session_state['quantiles_data']:
           st.session_state['quantiles_data'][current_variable_name] = {'pos':num}
        quantile_name = st.text_input(f"Enter Variable {num}'s Name: ",
                                        current_variable_name, 
                                        on_change=update_name,
                                        args=(num,),
                                        key=f"Quantile Name {num}")           
        if not all([x != '' for x in quantile_name]):
          st.warning('All variables must be named.')
          st.stop()
        #Fill in saved data
        print("quantile_name",len(quantile_name))

        if 'quantiles_data' in st.session_state and quantile_name in st.session_state['quantiles_data'] and 'number_of_quantiles' in st.session_state['quantiles_data'][quantile_name]:
            st.session_state[f'Quantile{num}'] = st.session_state['quantiles_data'][quantile_name]['number_of_quantiles']
            if 'q_data' in st.session_state['quantiles_data'][quantile_name]:
                print("writing q-data")
                for j,y in enumerate(st.session_state['quantiles_data'][quantile_name]['q_data'].reset_index().to_numpy()):
                    st.session_state[f"y values {quantile_name} {num*(j+1)}"] = y[0]
                    st.session_state[f"x values {quantile_name} {num*(j+1)}"] = y[1]
            if 'col_terms' in st.session_state['quantiles_data'][quantile_name]:
                st.session_state[f"Column_Terms {quantile_name} {quantile_index}"] = st.session_state['quantiles_data'][quantile_name]['col_terms']
            if 'boundedness' in st.session_state['quantiles_data'][quantile_name]:
                if st.session_state['quantiles_data'][quantile_name]['boundedness'] == 'b':
                    st.session_state[f"Column_Terms {quantile_name} {quantile_index}"] = 3
                    #convert to float and list
                    st.session_state["Column_lower"] = st.session_state['quantiles_data'][quantile_name]['bounds'][0]
                    # seed_container.number_input('Lower Bound')
                    st.session_state["Column_upper"] = st.session_state['quantiles_data'][quantile_name]['bounds'][1]
                elif st.session_state['quantiles_data'][quantile_name]['boundedness'] == 'sl':
                    st.session_state[f"Column_Terms {quantile_name} {quantile_index}"] = 1
                    #convert to float and list
                    st.session_state["Column_lower"] = st.session_state['quantiles_data'][quantile_name]['bounds'][-1]
                elif st.session_state['quantiles_data'][quantile_name]['boundedness'] == 'su':
                    st.session_state[f"Column_Terms {quantile_name} {quantile_index}"] = 2
                    # seed_container.number_input('Lower Bound')
                    st.session_state["Column_upper"] = st.session_state['quantiles_data'][quantile_name]['bounds'][-1]
                else: 
                    st.session_state[f"Column_Terms {quantile_name} {quantile_index}"] = 0
                     
                if 'seeds' in st.session_state['quantiles_data'][quantile_name]:    
                    st.session_state[f"entity {quantile_name}"] = st.session_state['quantiles_data'][quantile_name]['seeds']['arguments']['entity']
                    st.session_state[f"varId {quantile_name}"] = st.session_state['quantiles_data'][quantile_name]['seeds']['arguments']['varId']
                    st.session_state[f"seed3 {quantile_name}"] = st.session_state['quantiles_data'][quantile_name]['seeds']['arguments']['seed3']
                    st.session_state[f"seed4 {quantile_name}"] = st.session_state['quantiles_data'][quantile_name]['seeds']['arguments']['seed4']

        number_of_quantiles =  int(st.selectbox('Number of Quantiles', 
                                                                      range(3,6), 
                                                                      on_change = update_max_term,
                                                                      args = (num,quantile_name),
                                                                      key = f'Quantile{num}'))
        st.session_state['quantiles_data'][quantile_name]['number_of_quantiles'] = number_of_quantiles
        print("number_of_quantiles",number_of_quantiles)
        st.subheader("Enter Values Below:")
        terms_list = list(range(3,number_of_quantiles+1))
        st.session_state['quantiles_data'][quantile_name]['terms_list'] = terms_list
        y_values, x_values = st.columns(2)
        q_data = pd.DataFrame([[float(y_values.number_input(f'Percentage {i}',
                                        value = reference_probabilities[number_of_quantiles][i - 1] ,
                                        format = "%f",
                                        min_value = 0.0,
                                        max_value = 1.0,
                                        on_change = update_values,
                                        args = (quantile_name,f"y values {quantile_name} {num*i}", i - 1),
                                        key=f"y values {quantile_name} {num*i}")),
                                float(x_values.number_input(f'Value {i}', 
                                       value = 0.0,
                                       format = "%f",
                                       # step = 0.1
                                        on_change = update_values,
                                        args = (quantile_name,f"x values {quantile_name} {num*i}", i - 1),
                                        key=f"x values {quantile_name} {num*i}"))] for i in range(1,number_of_quantiles+1)],columns=['',quantile_name]).set_index('')   
        if 'q_data' in st.session_state['quantiles_data'][quantile_name]:
          if q_data[quantile_name].any():
            st.session_state['quantiles_data'][quantile_name]['q_data'] = q_data       
          else:
            st.session_state['quantiles_data'][quantile_name]['q_data'] = q_data
        else:
          st.session_state['quantiles_data'][quantile_name]['q_data'] = q_data
        col_terms = st.selectbox(f'Current Variable: Number of Terms', 
                            terms_list, 
                            index=len(terms_list) - 1,
                            on_change=update_terms, 
                            args=(quantile_name,data_type_str,quantile_index), 
                            key=f"Column_Terms {quantile_name} {num}")
        st.session_state['quantiles_data'][quantile_name]['col_terms'] = col_terms
        boundedness = st.selectbox(f'Current Variable: Boundedness', ("'u' - unbounded", 
                                                   "'sl' - semi-bounded lower", 
                                                   "'su' - semi-bounded upper",
                                                   "'b' - bounded on both sides"),
                                                   on_change=update_boundedness,
                                                   kwargs = dict(refresh = True, 
                                                       data_type = data_type_str, 
                                                       quantile_count = quantile_index, 
                                                       variable_name=quantile_name),
                                                   key=f"Column_boundedness {quantile_name} {num}")
        if boundedness == "'b' - bounded on both sides":
            #convert to float and list
            if "Column_lower" in st.session_state:
              st.session_state["Column_lower"] =  st.session_state["Column_lower"] if st.session_state["Column_lower"] <= q_data[quantile_name].min() else q_data[quantile_name].min() 
            boundsl = st.number_input('Lower Bound', 
                                                            max_value = q_data[quantile_name].min(),
                                                            value = q_data[quantile_name].min(),
                                                            on_change=update_boundedness,
                                                            kwargs = dict(refresh = True, 
                                                               data_type = data_type_str, 
                                                               quantile_count = quantile_index, 
                                                               variable_name=quantile_name),
                                                            key=f"Column_lower") 
            # seed_container.number_input('Lower Bound')
            if "Column_upper" in st.session_state:
              st.session_state["Column_upper"] =  st.session_state["Column_upper"] if st.session_state["Column_upper"] >= q_data[quantile_name].max() else q_data[quantile_name].max() 
            boundsu = st.number_input('Upper Bound', 
                                                            min_value = q_data[quantile_name].max(),
                                                            value = q_data[quantile_name].max(),
                                                            on_change=update_boundedness,
                                                            kwargs = dict(refresh = True, 
                                                               data_type = data_type_str, 
                                                               quantile_count = quantile_index, 
                                                               variable_name=quantile_name),
                                                            key=f"Column_upper")
            # seed_container.number_input('Upper Bound')
            bounds = [float(boundsl),float(boundsu)]
        elif boundedness.find("lower") != -1:
            if "Column_lower" in st.session_state:
              st.session_state["Column_lower"] =  st.session_state["Column_lower"] if st.session_state["Column_lower"] <= q_data[quantile_name].min() else q_data[quantile_name].min() 
            boundsl = st.number_input('Lower Bound', 
                                                            max_value = q_data[quantile_name].min(),
                                                            value = q_data[quantile_name].min(),
                                                            on_change=update_boundedness,
                                                            kwargs = dict(refresh = True, 
                                                               data_type = data_type_str, 
                                                               quantile_count = quantile_index, 
                                                               variable_name=quantile_name),
                                                            key=f"Column_lower") 
            bounds = [float(boundsl)]
        elif boundedness.find("upper") != -1:
            print("q_data[quantile_name].max()",q_data[quantile_name].max(),)
            if "Column_upper" in st.session_state:
              st.session_state["Column_upper"] =  st.session_state["Column_upper"] if st.session_state["Column_upper"] >= q_data[quantile_name].max() else q_data[quantile_name].max() 
            boundsu = st.number_input('Upper Bound', 
                                                            min_value = q_data[quantile_name].max(),
                                                            # value = q_data[quantile_name].max(),
                                                            on_change=update_boundedness,
                                                            kwargs = dict(refresh = True, 
                                                               data_type = data_type_str, 
                                                               quantile_count = quantile_index, 
                                                               variable_name=quantile_name),
                                                            key=f"Column_upper")
            bounds = [float(boundsu)]
        else:
            bounds = [0,1]      
        st.session_state['quantiles_data'][quantile_name]['bounds'] = bounds
        # if "selected_column" in locals():
            # print('selected column is here!')
            # print("selected_column",selected_column)
            # if 'mfitted' in st.session_state  and selected_column in st.session_state['mfitted'][data_type_str] :
                # print("selected_column", selected_column)
                # st.session_state["Column_Terms"] = st.session_state['mfitted'][data_type_str][selected_column]['options']['terms'] if st.session_state['mfitted'][data_type_str][selected_column]['options']['terms'] else number_of_quantiles
            
        boundedness = boundedness.strip().split(" - ")[0].replace("'","")
        st.session_state['quantiles_data'][quantile_name]['boundedness'] = boundedness
        # print('st.session_state["Column_Terms"]',st.session_state["Column_Terms"])
        seed_container = st.container()
        seed_container.write("Enter HDR Seed Values Below:")
        left_values, right_values = seed_container.columns(2)
        seed_data = [left_values.text_input(f'entity',value=0,
                                                                on_change=update_seeds,
                                                                kwargs=dict(data_type=data_type_str, variable_name=quantile_name),
                                                                key=f"entity {quantile_name}"),
                            right_values.text_input(f'varId',
                                                                value=num,
                                                                on_change=update_seeds,
                                                                kwargs=dict(data_type=data_type_str, variable_name=quantile_name),
                                                                key=f"varId {quantile_name}"),
                            left_values.text_input(f'seed 3',
                                                                value=0,
                                                                on_change=update_seeds,
                                                                kwargs=dict(data_type=data_type_str, variable_name=quantile_name),
                                                                key=f"seed3 {quantile_name}"),
                            right_values.text_input(f'seed 4',
                                                                value=0,
                                                                on_change=update_seeds,
                                                                kwargs=dict(data_type=data_type_str, variable_name=quantile_name),
                                                                key=f"seed4 {quantile_name}")]  
        st.session_state['quantiles_data'][quantile_name]['seeds'] = {
                                   'name':'hdr'+str(num),
                                   'function':'HDR_2_0',
                                   'arguments': {'counter':'PM_Index',
                                                        'entity' : seed_data[0],
                                                        'varId' :  seed_data[1],
                                                        'seed3' :  seed_data[2],
                                                        'seed4' :  seed_data[3] }}                
        previous_button = left_values.button("Previous Variable",on_click = update_counter,args=(-1,),key=f"previous_button")
        next_button = right_values.button("Next Variable",on_click = update_counter,args=(1,),key=f"next_button")                    
        make_graphs_button = left_values.button("Make Graph Panel",key=f"make_graph_panel_button{num}")
        # run_calculations = right_values.button("Run Calculations",key=f"run_calculations_button{num}")
        run_calculations = True
        left_values.markdown("")
        right_values.markdown("")
        st.session_state['quantiles_data'][quantile_name]['make_graphs_button'] = make_graphs_button
        st.session_state['quantiles_data'][quantile_name]['run_calculations'] = run_calculations
        if st.session_state['quantiles_data'][quantile_name]['make_graphs_button'] or st.session_state['quantiles_data'][quantile_name]['run_calculations']:
          st.session_state["Big Graph Column"] = quantile_name
            #add SPT normal or three terms                                                                
            # update_seeds(data_type_str)                                         
            # if num < quantile_number_variable:
        
        # if num > 1 and any(q_data[-1]):
            # quanile_container.error(f"enter a number greater zero for Value {num}.")
        # Add check that items are less than the other value percentage
        # if len(q_data) > 1:
            # q_data
    # with quanile_container.expander("Correlation Matrix Input"):     
        # st.write('sodm')
    print("data",st.session_state['quantiles_data'])
    # pd_data = pd.DataFrame(q_data,columns=['',*quantile_name]).set_index('')
    pd_data = pd.concat([st.session_state['quantiles_data'][x[1]]['q_data'] for x in enumerate(st.session_state['quantiles_data']) if x[0] < quantile_number_variable and 'q_data' in st.session_state['quantiles_data'][x[1]]])
    print(pd_data)
    if isinstance(pd_data, pd.DataFrame):
        st.session_state["column_index"] = {k:v for v,k in enumerate(pd_data.columns)}
            
    # st.subheader("Preview of Quantile Data")
    # st.write(pd_data.to_html(index=False), unsafe_allow_html=True)
    # boundedness = quanile_container.selectbox('Boundedness', 
                                                      # ("'u' - unbounded", 
                                                       # "'sl' - semi-bounded lower", 
                                                       # "'su' - semi-bounded upper",
                                                       # "'b' - bounded on both sides"),
                                                       # key=f"Quantile_boundedness")
                                                       
    # if boundedness == "'b' - bounded on both sides":
        # # convert to float and list
        # boundsl = y_values.text_input('Lower Bound', '0',key=f"Quantile_lower")
        # boundsu = x_values.text_input('Upper Bound', '1',key=f"Quantile_upper")
        # bounds = [float(boundsl),float(boundsu)]
    # elif boundedness.find("lower") != -1:
        # bounds = [float(y_values.text_input('Lower Bound', '0',key=f"Quantile_lower"))]
    # elif boundedness.find("upper") != -1:
        # bounds = [float(x_values.text_input('Upper Bound', '1',key=f"Quantile_upper"))]
    # else:
        # bounds = [0,1]
        
    # boundedness = boundedness.strip().split(" - ")[0].replace("'","")
    # big_plots = quanile_container.checkbox("Big Graphs?")
            # big_plots = quanile_container.checkbox("Big Graphs?",value=False)
    # if not "selected_column" in locals():
        # empty_table.empty()
    # selected_column = graphs_container.selectbox("Select Current Variable:",  pd_data.columns, key="Big Graph Column")     
    st.session_state["Big Graph Column"] = st.session_state[f"Quantile Name {st.session_state['quantile_counter']}"]
    variable_list =  [x for x in pd_data.columns]
    if "Big Graph Column" in st.session_state and not st.session_state["Big Graph Column"] in variable_list:
      variable_list.append(st.session_state["Big Graph Column"])
    print('variable_list',variable_list)
    selected_column = graphs_container.selectbox("Select Current Variable:", variable_list ,on_change=update_input_name, key="Big Graph Column") 
    # upper_bound, lower_bound = seed_container.columns(2)
    #TODO: error handling for value too small for upper bounds or letter         
    # if not run_calculations:
        # graph_panel = True
    # else:
        # graph_panel = False
    # if make_graphs_button:
    # run_calculations = True
        
    big_plots = True
    if st.session_state['quantiles_data'][selected_column]['run_calculations'] or st.session_state['quantiles_data'][selected_column]['make_graphs_button']:
         if big_plots:
                col_terms = int(st.session_state['quantiles_data'][selected_column]['col_terms'] ) if isinstance(st.session_state['quantiles_data'][selected_column]['col_terms'],int) else None
                if col_terms and pd_data[selected_column].any():
                    # if 'mfitted' in st.session_state and selected_column in st.session_state['mfitted'][data_type_str]:
                        # update_boundedness(refresh = True,data_type = data_type_str, quantile_count = st.session_state["column_index"][selected_column])
                    print(pd_data.loc[pd_data[selected_column].notnull(),[selected_column]])
                    print(pd_data[[selected_column]])  
                    pd_data.loc[pd_data[selected_column].notnull(),[selected_column]].apply(make_csv_graph,
                                                   probs = pd_data[pd_data[selected_column].notnull()].index.to_list(),
                                                   boundedness = st.session_state['quantiles_data'][selected_column]['boundedness'] ,
                                                   bounds = st.session_state['quantiles_data'][selected_column]['bounds'] ,
                                                   big_plots = big_plots,
                                                   user_terms = col_terms,
                                                   graphs = st.session_state['quantiles_data'][selected_column]['make_graphs_button'])
                    st.session_state['mfitted'][data_type_str][selected_column]['options']['seeds'] = st.session_state['quantiles_data'][selected_column]['seeds']
                    # if 'mfitted' in st.session_state and selected_column in st.session_state['mfitted'][data_type_str] and 'seeds' not in st.session_state['mfitted'][data_type_str][selected_column]['options']:
                        # st.session_state['mfitted'][data_type_str][selected_column]['options']['seeds'] = {
                                                                                                   # 'name':'hdr'+str(st.session_state["column_index"][selected_column]+1),
                                                                                                   # 'function':'HDR_2_0',
                                                                                                   # 'arguments': {'counter':'PM_Index',
                                                                                                                        # 'entity' : "0" ,
                                                                                                                        # 'varId' : "1",
                                                                                                                        # 'seed3' : "0",
                                                                                                                        # 'seed4' : "0" }}
                        # print('assigned seeds')
                    # elif 'mfitted' in st.session_state and selected_column in st.session_state['mfitted'][data_type_str] and f"entity {selected_column}" in st.session_state and st.session_state[f"entity {selected_column}"] == "":
                        # st.session_state['mfitted'][data_type_str][selected_column]['options']['seeds']['arguments'] = {'entity' : st.session_state[f"entity {selected_column}"] ,
                                                                                                    # 'varId' : st.session_state[f"varId {selected_column}"] ,
                                                                                                    # 'seed3' : st.session_state[f"seed3 {selected_column}"],
                                                                                                    # 'seed4' : st.session_state[f"seed4 {selected_column}"] }                      
                    # st.session_state[f"entity {selected_column}"] = str(st.session_state['mfitted'][data_type_str][selected_column]['options']['seeds']['arguments']["entity"])
                    # st.session_state[f"varId {selected_column}"] = str(st.session_state['mfitted'][data_type_str][selected_column]['options']['seeds']['arguments']["varId"])
                    # st.session_state[f"seed3 {selected_column}"] = str(st.session_state['mfitted'][data_type_str][selected_column]['options']['seeds']['arguments']["seed3"])
                    # st.session_state[f"seed4 {selected_column}"]  = str(st.session_state['mfitted'][data_type_str][selected_column]['options']['seeds']['arguments']["seed4"])  
                    # print(f"seed_data is {seed_data}")
               
        # pass
    input_data("Unknown",0,pd_data,pd_data.index.to_list())