import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import community as community_louvain
from networkx.algorithms import community
import itertools
from dateutil.parser import parse
from sklearn.preprocessing import MinMaxScaler as MMS
import os
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
import random
import plotly.graph_objects as go
import plotly.express as px
import yaml
from yaml import Loader


def update_counter(curr_counter, configs):
    configs["curr_counter"] = curr_counter
    yaml.dump(configs, open('configs.yaml', 'w'))
    return



#creates dataset for an ETF to be used by ElasticNet
def create_dataset(etf_set, ticker):
    train_df = etf_set.copy().drop(["Date"], axis=1).iloc[0:-1,:]
    Y = (etf_set[ticker].values[1:] - etf_set[ticker].values[0:-1])*pow(10,3)
    train_df["Y"] = etf_set[ticker].values[1:]
    train_df = train_df.dropna()
    X = train_df.drop("Y", axis=1).values
    col_list = list(train_df.drop("Y", axis=1).columns)
    Y = train_df["Y"].values 
    return(X, Y, col_list)

#extracts coefficients for elastic net and the corresponding connections
def elastic_net_connections(etf_set, ticker):
    X,Y,col_list = create_dataset(etf_set, ticker)
    x_train,x_val,y_train,y_val = train_test_split(X, Y, test_size = 0.25, shuffle = True)
    en_model = ElasticNet(l1_ratio=1,
                      fit_intercept = False,
                      normalize = True,
                      precompute = False)
    en_model.fit(x_train,y_train)
    score = en_model.score(x_val,y_val)
    #print("Fit linear model with score : {}".format(score))
    coefs = np.array(en_model.coef_).reshape(-1,1)
    coefs[col_list.index("timedelta")] = 0
    coefs[col_list.index(ticker)] = 0
    conn_idx = np.where(coefs!=0)[0]
    return([col_list[i] for i in conn_idx],
           [coefs[i][0] for i in conn_idx])

#converts partition in list form to dictionary
def dict_to_list(partition_map):
    partition = {}
    for node, p in partition_map.items():
        if p not in partition:
            partition[p] = [node]
        else:
            partition[p].append(node)
    partition_list = list(partition.values())
    return(partition_list)

#concerts partition in dictionary form to list of lists
def list_to_dict(partition_list):
    partition_map = {}
    for p, node_list in enumerate(partition_list):
        for node in node_list:
            partition_map[node] = p
    return(partition_map)

#applies louvian partition
def apply_louvain(UG):
    partition_map = community_louvain.best_partition(UG)
    partition_list = dict_to_list(partition_map)
    try:
        mod = nx.community.quality.modularity(UG, partition_list)
    except:
        mod = 0
    return mod, partition_map, partition_list

#applies lpa
def apply_lpa(UG):
    partition_list = list(community.asyn_lpa_communities(UG))
    partition_map = list_to_dict(partition_list)
    try:
        mod = nx.community.quality.modularity(UG, partition_list)
    except:
        mod = 0
    return mod, partition_map, partition_list

#applies label propagation
def apply_label(UG):
    partition_list = list(community.label_propagation_communities(UG))
    partition_map = list_to_dict(partition_list)
    try:
        mod = nx.community.quality.modularity(UG, partition_list)
    except:
        mod = 0
    return mod, partition_map, partition_list

def apply_none(UG):
    return(None, None, None)

def construct_graph(etf_set, method, cutoff=None):
    group_tickers = list(set(etf_set.columns) - set(["Date","timedelta"]))
    G = nx.Graph()
    G.add_nodes_from(group_tickers)
    if(method=="elasticnet"):
        all_weights = []
        for ticker in group_tickers:
            connections, weights = elastic_net_connections(etf_set, ticker)
            for i,conn_ticker in enumerate(connections):
                if(weights[i]!=0):
                    if(~G.has_edge(ticker, conn_ticker)):
                        G.add_edge(ticker, conn_ticker, weight = weights[i])
                    else:
                        G.edges()[ticker, conn_ticker]['weight'] = (G.edges()[ticker, conn_ticker]['weight'] + weights[i])/2
                    all_weights.append(weights[i])
        min_weight, max_weight = np.amin(all_weights), np.amax(all_weights)
        for i in list(G.edges()):
            adj_weight = (G.edges()[i]['weight']-min_weight) / (max_weight - min_weight)
            G.edges()[i]['weight'] = adj_weight
    else:
        for (etf1,etf2) in list(itertools.combinations(group_tickers, 2)):
            c = np.abs(etf_set[[etf1,etf2]].dropna().corr().values[0,1])
            if(c > cutoff):
                G.add_edge(etf1, etf2, weight = c)
    return(G)


def partitioned_plotly_graph(G, pos, title, partition_map = None):
    
    ##Creating edges
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]][0],pos[edge[0]][1]
        x1, y1 = pos[edge[1]][0],pos[edge[1]][1]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')
    
    ##Creating nodes
    node_trace_df = pd.DataFrame()
    node_trace_df["node"] = list(G.nodes())
    node_trace_df["x"] = node_trace_df["node"].apply(lambda node:pos[node][0])
    node_trace_df["y"] = node_trace_df["node"].apply(lambda node:pos[node][1])
    if(partition_map!=None):
        node_trace_df["partition"] = node_trace_df["node"].apply(partition_map.get).astype(str)
    node_trace_df["degree"] = node_trace_df["node"].apply(dict(G.degree).get)
    node_trace_df["size"] = 10
    
    if(partition_map!=None):
        node_traces = list(px.scatter(node_trace_df, x="x", y="y", color="partition",text="node",
                           hover_data=["node","degree","partition"],size="size").to_dict()['data'])
    else:
        node_traces = list(px.scatter(node_trace_df, x="x", y="y", color="degree",text="node",
                           hover_data=["node","degree"],size="size").to_dict()['data'])

    
    ##Main graph
    fig = go.Figure(data=[edge_trace] + node_traces,
             layout=go.Layout(
                title=title,
                titlefont_size=16,
                showlegend=True,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=True),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=True))
                )
    
    return(fig)


def calc_backbone(d1,d2,s1,s2,wt):
    prob1 = np.power((1 - (wt/s1)),(d1 - 1))
    prob2 = np.power((1 - (wt/s2)),(d2 - 1))
    return prob1 if prob1 <= prob2 else prob2

def get_backbone(G, cutoff):
    WG = G.copy()
    wgpos = nx.layout.spring_layout(WG)
    WG_weights = np.real([*nx.get_edge_attributes(WG, 'weight').values()])
    degree = dict(WG.degree())
    strength = dict(WG.degree(weight='weight'))
    edges = WG.edges()
    deg_seq = list(degree.values())
    str_seq = list(strength.values())
    # Calculate the bweight probability for the edges using backbone formula and prepare the edge list to remove
    edge_prob = {}
    for n1,n2, wt in WG.edges.data("weight"):
        edge_prob[(n1,n2)] = calc_backbone(degree[n1],degree[n2],strength[n1],strength[n2],wt)     
    edges_to_remove = [k for k,v in edge_prob.items() if v > cutoff]
    WG.remove_edges_from(edges_to_remove)
    return(WG)