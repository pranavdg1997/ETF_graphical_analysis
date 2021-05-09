import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import os
from utils import *
from dateutil.parser import parse
import streamlit as st
from PIL import Image
import yaml
from yaml import Loader
import plotly
import json
st.set_page_config(layout="wide")
# st.beta_set_page_config(layout="wide")
st.title("SP21: NETWORK SCIENCE: 10075 Project Presentation")

configs = yaml.load(open("configs.yaml"),
    Loader=Loader)

try:
    os.mkdir("jsons")
except:
    pass

etf_set = pd.read_csv(configs["dataset"])
etf_set["Date"] = pd.to_datetime(etf_set["Date"])
etf_set.sort_values(by="Date", inplace=True)
dt0 = parse(etf_set["Date"].astype(str).values[0])
etf_set["timedelta"] = etf_set["Date"].apply(lambda d:(parse(str(d))-dt0).days)



# st.title('Link Analysis of ETF data')
# st.write('This app allows you to take a deeper look at graphical analysis of ETF( Exchange Traded Funds), and interactively explore relationships.')

# col1, col2 = st.beta_columns((1,3))
# reset_button = col1.button(key="reset_button",
#         label="Start slides")
# prev_button = col1.button(key="prev_button",
#         label="Previous Slide")
# next_button = col1.button(key="next_button",
#         label="Next Slide")


# curr_counter = yaml.load(open("configs.yaml"),
#     Loader=Loader)["curr_counter"]
# max_limit = 1
# #col2.image(Image.open("slides/{}.JPG".format(curr_counter)), use_column_width=True)




# if(prev_button|next_button|reset_button):
#     if(prev_button):
#         curr_counter = max(0, curr_counter-1)
#         update_counter(curr_counter, configs)
#     elif(next_button):
#         curr_counter = min(curr_counter + 1, max_limit)
#         update_counter(curr_counter, configs)
#     else:
#         curr_counter = 0
#         update_counter(curr_counter, configs)
# col2.image(Image.open("slides/{}.JPG".format(curr_counter)), use_column_width=True)



col_corr, col_en = st.beta_columns((1,1))

#Pearson part
with col_corr:
    col_corr.markdown("## Pearson Correlation Analysis")
    cutoff = col_corr.slider("Correlation Cutoff", 
        min_value=0.5,
        max_value=1.0,
        value=0.8, step=0.05)

    

    with st.spinner('Wait for it...'):  
                

        partition_method = col_corr.selectbox("Partition Method", ["none","louvian","lpa","lp"], key="Pearson")
        if(partition_method =="louvian"):
            pmf_corr = apply_louvain
        elif(partition_method =="lpa"):
            pmf_corr= apply_lpa
        elif(partition_method == "lp"):
            pmf_corr = apply_label
        else:
            pmf_corr = apply_none

        extract_backbone = col_corr.checkbox("Extract backbone?", value=False, key="Pearson", help=None)

        if(extract_backbone):
            backbone_cutoff = col_corr.slider("Backbone Cutoff", 
                min_value=0.05,
                max_value=0.5,
                value=0.25, step=0.05)
            graph_name = "{}_{}_{}_{}_{}.json".format("pearson",
                cutoff,
                partition_method,
                extract_backbone,
                backbone_cutoff)  
        else:
            graph_name = "{}_{}_{}_{}.json".format("pearson",
                cutoff,
                partition_method,
                extract_backbone)

        title = "ETF Relationships (method:{}|partition_method:{}|extract_backbone:{})".format("pearson",
                                                                                              str(partition_method),
                                                                                              str(extract_backbone))

        if(graph_name in os.listdir("jsons")):
            fig_corr = plotly.io.from_json(open(os.path.join("jsons/",graph_name)).read())
        else:
            G_corr = construct_graph(etf_set,
                            method="pearson",
                            cutoff = cutoff)
            if(extract_backbone):
                G_corr = get_backbone(G_corr, backbone_cutoff)                                                            
            pos = nx.spring_layout(G_corr,
                                   weight='weight')
            mod, partition_map, partition_list = pmf_corr(G_corr.copy())
            fig_corr = partitioned_plotly_graph(G_corr,
                                     pos,
                                     partition_map=partition_map,
                                      title = title)
            plotly.io.write_json(fig_corr,os.path.join("jsons",graph_name))

        plotly_corr = col_corr.plotly_chart(fig_corr,
                use_column_width=True)


#ElasticNet part
with col_en:
    col_en.markdown("## ElasticNet Correlation Analysis")

    

    with st.spinner('Wait for it...'):  
        
        

        partition_method_en = col_en.selectbox("Partition Method", ["none","louvian","lpa","lp"], key="ElasticNet")
        if(partition_method =="louvian"):
            pmf_en = apply_louvain
        elif(partition_method =="lpa"):
            pmf_en= apply_lpa
        elif(partition_method == "lp"):
            pmf_en = apply_label
        else:
            pmf_en = apply_none

        extract_backbone_en = col_en.checkbox("Extract backbone?", value=False, key="ElasticNet", help=None)

        if(extract_backbone_en):
            backbone_cutoff_en = col_en.slider("Backbone Cutoff", 
                min_value=0.05,
                max_value=0.5,
                value=0.25, step=0.05)
            graph_name = "{}_{}_{}_{}_{}.json".format("elasticnet",
                cutoff,
                partition_method_en,
                extract_backbone_en,
                backbone_cutoff_en)  
        else:
            graph_name = "{}_{}_{}_{}.json".format("elasticnet",
                cutoff,
                partition_method_en,
                extract_backbone_en)

        title = "ETF Relationships (method:{}|partition_method:{}|extract_backbone:{})".format("pearson",
                                                                                              str(partition_method_en),
                                                                                              str(extract_backbone_en))

        if(graph_name in os.listdir("jsons")):
            fig_en = plotly.io.from_json(open(os.path.join("jsons/",graph_name)).read())
        else:
            G_en = construct_graph(etf_set,
                            method="elasticnet")
            if(extract_backbone_en):
                G_en = get_backbone(G_en, backbone_cutoff_en)                                                                              
            pos = nx.spring_layout(G_en,
                                   weight='weight')
            mod, partition_map, partition_list = pmf_en(G_en.copy())
            fig_en = partitioned_plotly_graph(G_en,
                                     pos,
                                     partition_map=partition_map,
                                     title = title)
            plotly.io.write_json(fig_en,os.path.join("jsons",graph_name))

        plotly_en = col_en.plotly_chart(fig_en,
                use_column_width=True)
