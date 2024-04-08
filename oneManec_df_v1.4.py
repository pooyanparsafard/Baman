#-------------------------------------------- imports
import pandas as pd 
# import numpy as np
import dash_daq as daq
from dash import Dash, dcc, html, Input, Output, dash_table, State
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sqlalchemy import create_engine
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import psycopg2
from itertools import combinations
from collections import Counter
import networkx as nx
import ast


#
def query_to_dataframe(query, connection):
    """
    Execute a SQL query and return the result as a pandas DataFrame
    """
    try:
        cursor = connection.cursor()
        cursor.execute(query)
        columns = [desc[0] for desc in cursor.description]
        result = cursor.fetchall()
        dataframe = pd.DataFrame(result, columns=columns)
        return dataframe
    except (Exception, psycopg2.Error) as error:
        print("Error while fetching data from PostgreSQL:", error)
        return None

# Connect to PostgreSQL database
try:
    username = 'report_user_read'
    password_with_at = 'FmmC53@nBARx*gG+E#*4q9^f_7'
    hostname = '10.1.10.82'
    port = '5678'
    database_name = 'report'
    connection = psycopg2.connect(user=username,
                                  password=password_with_at,
                                  host=hostname,
                                  port=port,
                                  database=database_name)
    print("Connected to PostgreSQL")
except (Exception, psycopg2.Error) as error:
    print("Error while connecting to PostgreSQL:", error)

# # Define your SQL query
# sql_query = 'SELECT * FROM "OneManexKnowledges" omk  LIMIT 100;'

# # Execute query and get DataFrame
# dataframe = query_to_dataframe(sql_query, connection)
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------- sep Dataframe
## sep Dataframe
# sep = query_to_dataframe('SELECT * FROM "OneManexKnowledges" omk  LIMIT 300000;', connection)
sep = query_to_dataframe('WITH last_inserted_date AS ( SELECT MAX("TransactionDateTime") AS last_inserted_date FROM "OneManexKnowledges" ), last_days_records AS (SELECT * FROM "OneManexKnowledges" WHERE "TransactionDateTime" >= (SELECT last_inserted_date FROM last_inserted_date) - INTERVAL \'1 days\') SELECT * FROM last_days_records;', connection)


#-------------------------------------------- terminal Dataframe
## terminal Dataframe
terminal = query_to_dataframe('select "BankTerminalId"  from "PartnerBranchTerminals" where "IsDeleted" = false;', connection)
terminals = [str(value) for value in set(terminal['BankTerminalId'])]

# #-------------------------------------------- card
# ## Table UserCards Query (func)
# def query_UserCards_daily():
#     engine = create_engine(connection_url)
#     with engine.connect() as connection:
#         result = connection.execute('select * from "UserCards" uc where "IsDelete" = false limit 10;')
#         df =  pd.DataFrame(result)
#         return (df)

# ## card Dataframe
# card = query_UserCards_daily()
 
# ## Baman cards list
# cards = list(set(card['HashPan'].str.lower()))

# #-------------------------------------------- loc
# loc = pd.read_csv('./data/loc.csv')
loc = pd.read_csv('./loc.csv')
loc.drop('index', axis=1, inplace=True)
new_columns = {'ترمینال':'TerminalId', 'شعبه':'Branch', 'صنف':'Class', 'کد صنف':'Class_code'}
loc.rename(columns=new_columns, inplace=True)
loc['TerminalId'] = loc['TerminalId'].astype('str')

#-------------------------------------------- globals
grand_selection = 200

#-------------------------------------------- functions
def convert_number(num):
    if num >= 10**6:
        return f"{num / 10**6:.2f}M"
    elif num >= 10**3:
        return f"{num / 10**3:.0f}k"
    else:
        return num
    

#-------------------------------------------- sep pre-processing
## Initialization
sep.rename(columns={'Annotation':'ann'}, inplace=True)
sep = pd.merge(sep, loc, on='TerminalId', how='left')
sep['CardHash'] = sep['CardHash'].str.lower()
sep['TerminalId'] = sep['TerminalId'].astype('str')
sep['TotalAmount'] = sep['TotalAmount'].apply(lambda x : int(str(x)[:-1]))
sep['Class'] = sep['Class'].astype('str')
sep['Branch'] = sep['Branch'].astype('str')
sep['CardHash_'] = sep['CardHash'].apply(lambda x : x[:20])
sep['IsOurTerminal'] = sep['TerminalId'].isin(terminals)
sep['tr'] = sep['TotalAmount']
sep['tr_'] = sep['tr'].map(convert_number)

## adding Card Aggregation Parameters
sep['Card_TotTrsPerAllTerms'] = sep.groupby('CardHash')['TotalAmount'].transform('sum')
sep['Card_TotTrsPerOurTerms'] = sep[sep['IsOurTerminal']==True].groupby('CardHash')['TotalAmount'].transform('sum') 
sep['Card_TotTrsPerOurTerms'] = sep.groupby('CardHash')['Card_TotTrsPerOurTerms'].ffill()
sep['Card_TotTrsPerOurTerms'] = sep.groupby('CardHash')['Card_TotTrsPerOurTerms'].bfill()
sep['Card_TotTrsPerAllTerms_'] = sep['Card_TotTrsPerAllTerms'].map(convert_number)
sep['Card_TotTrsPerOurTerms_'] = sep['Card_TotTrsPerOurTerms'].apply(convert_number)

sep['Card_CountTrsPerAllTerms'] = sep.groupby('CardHash')['CardHash'].transform('count')
sep['Card_CountTrsPerAllTerms'] = sep['Card_CountTrsPerAllTerms'].astype('int')
sep['Card_CountTrsPerOurTerms'] = sep[sep['IsOurTerminal']==True].groupby('CardHash')['CardHash'].transform('count')
sep['Card_CountTrsPerOurTerms'] = sep.groupby('CardHash')['Card_CountTrsPerOurTerms'].ffill()
sep['Card_CountTrsPerOurTerms'] = sep.groupby('CardHash')['Card_CountTrsPerOurTerms'].bfill()

sep['Card_TrCountPerEachTerm'] = sep.groupby(['CardHash','TerminalId'])['TerminalId'].transform(lambda x: x.count())

sep['Card_DiscBetTotTrsOfAllAndOurTerms'] = sep['Card_TotTrsPerAllTerms'] - sep ['Card_TotTrsPerOurTerms']
sep['Card_DiscBetTotTrsOfAllAndOurTerms_'] = sep['Card_DiscBetTotTrsOfAllAndOurTerms'].map(convert_number)

sep['Card_TotTermsAreNotOurList'] = sep[sep['IsOurTerminal'] == False].groupby('CardHash')['TerminalId'].transform(lambda x: ','.join(x))
sep['Card_TotTermsAreNotOurList'] = sep.groupby('CardHash')['Card_TotTermsAreNotOurList'].ffill()
sep['Card_TotTermsAreNotOurList'] = sep.groupby('CardHash')['Card_TotTermsAreNotOurList'].bfill()
sep['Card_TotTermsAreNotOurSet'] = sep[sep['IsOurTerminal'] == False].groupby('CardHash')['TerminalId'].transform(lambda x: str(set(x)))
sep['Card_TotTermsAreNotOurSet'] = sep.groupby('CardHash')['Card_TotTermsAreNotOurSet'].ffill()
sep['Card_TotTermsAreNotOurSet'] = sep.groupby('CardHash')['Card_TotTermsAreNotOurSet'].bfill()
sep['Card_TotTermsAreOurList'] = sep[sep['IsOurTerminal'] == True].groupby('CardHash')['TerminalId'].transform(lambda x: ','.join(x))
sep['Card_TotTermsAreOurList'] = sep.groupby('CardHash')['Card_TotTermsAreOurList'].ffill()
sep['Card_TotTermsAreOurList'] = sep.groupby('CardHash')['Card_TotTermsAreOurList'].bfill()
sep['Card_TotTermsAreOurSet'] = sep[sep['IsOurTerminal'] == True].groupby('CardHash')['TerminalId'].transform(lambda x: str(set(x)))
sep['Card_TotTermsAreOurSet'] = sep.groupby('CardHash')['Card_TotTermsAreOurSet'].ffill()
sep['Card_TotTermsAreOurSet'] = sep.groupby('CardHash')['Card_TotTermsAreOurSet'].bfill()

sep['Card_TotClassesAreNotOurList'] = sep[sep['IsOurTerminal'] == False].groupby('CardHash')['Class'].transform(lambda x: ' | '.join(x))
sep['Card_TotClassesAreNotOurList'] = sep.groupby('CardHash')['Card_TotClassesAreNotOurList'].ffill()
sep['Card_TotClassesAreNotOurList'] = sep.groupby('CardHash')['Card_TotClassesAreNotOurList'].bfill()
sep['Card_TotClassesAreNotOurSet'] = sep[sep['IsOurTerminal'] == False].groupby('CardHash')['Class'].transform(lambda x: str(set(x)))
sep['Card_TotClassesAreNotOurSet'] = sep.groupby('CardHash')['Card_TotClassesAreNotOurSet'].ffill()
sep['Card_TotClassesAreNotOurSet'] = sep.groupby('CardHash')['Card_TotClassesAreNotOurSet'].bfill()
sep['Card_TotClassesAreOurList'] = sep[sep['IsOurTerminal'] == True].groupby('CardHash')['Class'].transform(lambda x: ' | '.join(x))
sep['Card_TotClassesAreOurList'] = sep.groupby('CardHash')['Card_TotClassesAreOurList'].ffill()
sep['Card_TotClassesAreOurList'] = sep.groupby('CardHash')['Card_TotClassesAreOurList'].bfill()
sep['Card_TotClassesAreOurSet'] = sep[sep['IsOurTerminal'] == True].groupby('CardHash')['Class'].transform(lambda x: str(set(x)))
sep['Card_TotClassesAreOurSet'] = sep.groupby('CardHash')['Card_TotClassesAreOurSet'].ffill()
sep['Card_TotClassesAreOurSet'] = sep.groupby('CardHash')['Card_TotClassesAreOurSet'].bfill()

sep['Card_TrsAreNotOurList'] = sep[sep['IsOurTerminal'] == False].groupby('CardHash')['tr_'].transform(lambda x: ','.join(x))
sep['Card_TrsAreNotOurList'] = sep.groupby('CardHash')['Card_TrsAreNotOurList'].ffill()
sep['Card_TrsAreNotOurList'] = sep.groupby('CardHash')['Card_TrsAreNotOurList'].bfill()
sep['Card_TrsAreOurList'] = sep[sep['IsOurTerminal'] == True].groupby('CardHash')['tr_'].transform(lambda x: ','.join(x))
sep['Card_TrsAreOurList'] = sep.groupby('CardHash')['Card_TrsAreOurList'].ffill()
sep['Card_TrsAreOurList'] = sep.groupby('CardHash')['Card_TrsAreOurList'].bfill()

sep['Card_TotBranchesAreNotOurList'] = sep[sep['IsOurTerminal'] == False].groupby('CardHash')['Branch'].transform(lambda x: ' | '.join(x))
sep['Card_TotBranchesAreNotOurList'] = sep.groupby('CardHash')['Card_TotBranchesAreNotOurList'].ffill()
sep['Card_TotBranchesAreNotOurList'] = sep.groupby('CardHash')['Card_TotBranchesAreNotOurList'].bfill()
sep['Card_TotBranchesAreNotOurSet'] = sep[sep['IsOurTerminal'] == False].groupby('CardHash')['Branch'].transform(lambda x: str(set(x)))
sep['Card_TotBranchesAreNotOurSet'] = sep.groupby('CardHash')['Card_TotBranchesAreNotOurSet'].ffill()
sep['Card_TotBranchesAreNotOurSet'] = sep.groupby('CardHash')['Card_TotBranchesAreNotOurSet'].bfill()
sep['Card_TotBranchesAreOurList'] = sep[sep['IsOurTerminal'] == True].groupby('CardHash')['Branch'].transform(lambda x: ' | '.join(x))
sep['Card_TotBranchesAreOurList'] = sep.groupby('CardHash')['Card_TotBranchesAreOurList'].ffill()
sep['Card_TotBranchesAreOurList'] = sep.groupby('CardHash')['Card_TotBranchesAreOurList'].bfill()
sep['Card_TotBranchesAreOurSet'] = sep[sep['IsOurTerminal'] == True].groupby('CardHash')['Branch'].transform(lambda x: str(set(x)))
sep['Card_TotBranchesAreOurSet'] = sep.groupby('CardHash')['Card_TotBranchesAreOurSet'].ffill()
sep['Card_TotBranchesAreOurSet'] = sep.groupby('CardHash')['Card_TotBranchesAreOurSet'].bfill()

## adding Terminal Aggregation Parameters
sep['Term_TotTrsTerm'] = sep.groupby('TerminalId')['TotalAmount'].transform('sum') 

sep['Term_TotTrsTerm_'] = sep['Term_TotTrsTerm'].apply(convert_number)

sep['Term_CountTrsPerTerm'] = sep.groupby('TerminalId')['TerminalId'].transform('count')
sep['Term_CountTrsPerTerm'] = sep['Term_CountTrsPerTerm'].astype('int')

sep['Term_CountCardsPerTerm'] = sep.groupby('TerminalId')['CardHash_'].transform(lambda x: len(set(x)))

## adding Class Aggregation Parameters
sep['Class_TotTrsClass'] = sep.groupby('Class')['TotalAmount'].transform('sum') 
sep['Class_TotTrsNotOurClass'] = sep[sep['IsOurTerminal'] == False].groupby('Class')['TotalAmount'].transform('sum') 
sep['Class_TotTrsNotOurClass'] = sep.groupby('Class')['Class_TotTrsNotOurClass'].ffill()
sep['Class_TotTrsNotOurClass'] = sep.groupby('Class')['Class_TotTrsNotOurClass'].bfill()
sep['Class_TotTrsOurClass'] = sep[sep['IsOurTerminal'] == True].groupby('Class')['TotalAmount'].transform('sum') 
sep['Class_TotTrsOurClass'] = sep.groupby('Class')['Class_TotTrsOurClass'].ffill()
sep['Class_TotTrsOurClass'] = sep.groupby('Class')['Class_TotTrsOurClass'].bfill()


sep['Class_TotTrsClass_'] = sep['Class_TotTrsClass'].apply(convert_number)
sep['Class_TotTrsNotOurClass_'] = sep['Class_TotTrsNotOurClass'].apply(convert_number)
sep['Class_TotTrsOurClass_'] = sep['Class_TotTrsOurClass'].apply(convert_number)


sep['Class_CountTrsPerClass'] = sep.groupby('Class')['Class'].transform('count')
sep['Class_CountTrsPerClass'] = sep['Class_CountTrsPerClass'].astype('int')

sep['Class_CountCardsPerClass'] = sep.groupby('Class')['CardHash_'].transform(lambda x: len(set(x)))

## adding Branch Aggregation Parameters
sep['Branch_TotTrsBranch'] = sep.groupby('Branch')['TotalAmount'].transform('sum') 

sep['Branch_TotTrsBranch_'] = sep['Branch_TotTrsBranch'].apply(convert_number)

sep['Branch_CountTrsPerBranch'] = sep.groupby('Branch')['Branch'].transform('count')
sep['Branch_CountTrsPerBranch'] = sep['Branch_CountTrsPerBranch'].astype('int')

sep['Branch_CountCardsPerBranch'] = sep.groupby('Branch')['CardHash_'].transform(lambda x: len(set(x)))

#-------------------------------------------- Extracted Dataframes
#### Parameter 'high_Card_DiscBetTotTrsOfAllAndOurTerms_df' --> 1 GEO & Scatter
columns_selection = ['CardHash_','Card_DiscBetTotTrsOfAllAndOurTerms_','Card_TotTrsPerAllTerms_', 'Card_TotTrsPerOurTerms_','Card_CountTrsPerAllTerms',
                     'Card_CountTrsPerOurTerms','Card_TotTermsAreNotOurList','Card_TotTermsAreOurList','Card_TrsAreNotOurList','Card_TrsAreOurList','Card_TotClassesAreNotOurList','Card_TotClassesAreOurList',
                     'Card_TotBranchesAreNotOurList','Card_TotBranchesAreOurList','Card_TotTermsAreNotOurSet','Card_TotTermsAreOurSet','Card_TotClassesAreNotOurSet','Card_TotClassesAreOurSet',
                     'Card_TotBranchesAreNotOurSet','Card_TotBranchesAreOurSet','CardHash']
high_Card_DiscBetTotTrsOfAllAndOurTerms_df = sep.sort_values(['Card_DiscBetTotTrsOfAllAndOurTerms','CardHash','tr'], ascending=False)
high_Card_DiscBetTotTrsOfAllAndOurTerms_plot = high_Card_DiscBetTotTrsOfAllAndOurTerms_df.drop_duplicates(subset = ['CardHash'])
high_Card_DiscBetTotTrsOfAllAndOurTerms_plot = high_Card_DiscBetTotTrsOfAllAndOurTerms_plot[high_Card_DiscBetTotTrsOfAllAndOurTerms_plot['Card_DiscBetTotTrsOfAllAndOurTerms'] > 0 ]
high_Card_DiscBetTotTrsOfAllAndOurTerms_table = high_Card_DiscBetTotTrsOfAllAndOurTerms_plot[columns_selection]
high_Card_DiscBetTotTrsOfAllAndOurTerms_table.rename(columns={'Card_DiscBetTotTrsOfAllAndOurTerms_':'Disc___', 'Card_TotTrsPerAllTerms_':'tr_tms__', 'Card_TotTrsPerOurTerms_':'tr_tms_B',
                                                               'Card_CountTrsPerAllTerms':'c__', 'Card_CountTrsPerOurTerms':'c_B', 'Card_TotTermsAreNotOurList':'tms__',
                                                               'Card_TotTermsAreOurList':'tms_B', 'Card_TrsAreNotOurList':'tr_ls__', 'Card_TrsAreOurList':'tr_ls_B',
                                                               'Card_TotClassesAreNotOurList':'cls_ls__', 'Card_TotClassesAreOurList':'cls_ls_B',
                                                               'Card_TotBranchesAreNotOurList':'br_ls__', 'Card_TotBranchesAreOurList':'br_ls_B',
                                                               'Card_TotTermsAreNotOurSet':'tms_com__', 'Card_TotTermsAreOurSet':'tms_com_B', 
                                                               'Card_TotClassesAreNotOurSet':'cls_com__', 'Card_TotClassesAreOurSet':'cls_com_B',
                                                               'Card_TotBranchesAreNotOurSet':'br_com__', 'Card_TotBranchesAreOurSet':'br_com_B' }, inplace=True)
min_Card_DiscBetTotTrsOfAllAndOurTerms = int(high_Card_DiscBetTotTrsOfAllAndOurTerms_df['Card_DiscBetTotTrsOfAllAndOurTerms'].min())
max_Card_DiscBetTotTrsOfAllAndOurTerms = int(high_Card_DiscBetTotTrsOfAllAndOurTerms_df['Card_DiscBetTotTrsOfAllAndOurTerms'].max())


#### Parameter 'high_Card_DiscBetTotTrsOfAllAndOurTerms_barchart' --> 1 Barcharts (grand_selection first Cards)
list_of_FirstGrandSelectionCards = list(high_Card_DiscBetTotTrsOfAllAndOurTerms_plot.iloc[:grand_selection]['CardHash_'])
high_Card_DiscBetTotTrsOfAllAndOurTerms_barchart = high_Card_DiscBetTotTrsOfAllAndOurTerms_df[high_Card_DiscBetTotTrsOfAllAndOurTerms_df['CardHash_'].isin(list_of_FirstGrandSelectionCards)]
# high_Card_DiscBetTotTrsOfAllAndOurTerms_barchart = high_Card_DiscBetTotTrsOfAllAndOurTerms_barchart.sort_values('Card_DiscBetTotTrsOfAllAndOurTerms', ascending=False)
min_high_Card_DiscBetTotTrsOfAllAndOurTerms_barchart=int(high_Card_DiscBetTotTrsOfAllAndOurTerms_barchart['Card_DiscBetTotTrsOfAllAndOurTerms'].min())
max_high_Card_DiscBetTotTrsOfAllAndOurTerms_barchart=int(high_Card_DiscBetTotTrsOfAllAndOurTerms_barchart['Card_DiscBetTotTrsOfAllAndOurTerms'].max())


#### Parameter 'high_Term_df' --> 2 GEO  --> 2000 terms
columns_selection = ['TerminalId','Class','Branch','Term_TotTrsTerm_','Term_CountTrsPerTerm','Term_CountCardsPerTerm','IsOurTerminal','Term_TotTrsTerm','lat','long']
high_Term_df = sep.sort_values(['Term_TotTrsTerm','TerminalId'], ascending=False)
high_Term_plot = high_Term_df.drop_duplicates(subset = ['TerminalId'])
high_Term_plot = high_Term_plot[columns_selection]
high_Term_plot = high_Term_plot.head(2000)
high_Term_plot['IsOurTerminal'].replace({True:'Yes', False:'No'}, inplace=True)
min_high_Term = int(high_Term_df['Term_TotTrsTerm'].min())
max_high_Term = int(high_Term_df['Term_TotTrsTerm'].max())

#### Parameter 'high_Class_df' 
columns_selection = ['Class','Class_TotTrsClass_','Class_TotTrsNotOurClass_','Class_TotTrsOurClass_','Class_CountTrsPerClass','Class_CountCardsPerClass','IsOurTerminal','Class_TotTrsClass','Class_TotTrsNotOurClass','Class_TotTrsOurClass']
high_Class_df = sep.sort_values(['Class_TotTrsClass','Class'], ascending=False)
high_Class_plot = high_Class_df.drop_duplicates(subset = ['Class'])
high_Class_plot = high_Class_plot[columns_selection]
min_high_Class = int(high_Class_df['Class_TotTrsClass'].min())
max_high_Class = int(high_Class_df['Class_TotTrsClass'].max())

#### Parameter 'high_Branch_df' --> 3 GEO
columns_selection = ['Branch','Branch_TotTrsBranch_','Branch_CountTrsPerBranch','Branch_CountCardsPerBranch','Branch_TotTrsBranch','lat','long']
high_Branch_df = sep.sort_values(['Branch_TotTrsBranch','Branch'], ascending=False)
high_Branch_plot = high_Branch_df.drop_duplicates(subset = ['Branch'])
high_Branch_plot = high_Branch_plot[columns_selection]
min_high_Branch = int(high_Branch_df['Branch_TotTrsBranch'].min())
max_high_Branch = int(high_Branch_df['Branch_TotTrsBranch'].max())


#### Parameter edges graph (Baman)
# graph df
selection = ['Class','Class_code','Class_TotTrsClass', 'Class_CountTrsPerClass', 'Class_CountCardsPerClass']
nodes_graph_df = high_Card_DiscBetTotTrsOfAllAndOurTerms_df [selection]
nodes_graph_df = nodes_graph_df.drop_duplicates('Class')

# getting nodes and edges
column_TotClassesAreOurSet = high_Card_DiscBetTotTrsOfAllAndOurTerms_df.drop_duplicates('CardHash_')['Card_TotClassesAreOurSet']
column_TotClassesAreOurSet = column_TotClassesAreOurSet[pd.notna(column_TotClassesAreOurSet)]
list_TotClassesAreOurSet = list(column_TotClassesAreOurSet)
list_TotClassesAreOurSet = [ast.literal_eval(x) for x in list_TotClassesAreOurSet]
c = Counter()
nodes=set()
for row_set in list_TotClassesAreOurSet:
    row_list = list(row_set)
    nodes.update(row_set)
    comb = combinations(row_list, 2)
    c.update(Counter(comb))

edges = [(k[0], k[1], v) for k,v in c.most_common(20)]   ###----------->  esdges constraint
nodes = list(nodes)
nodes = list(set([x for x in nodes for y in edges if x==y[0] or x == y[1]]))  ###----------->  nodes constraint
nodes_graph_df['node_positions'] = nodes_graph_df.apply(lambda x: f"{x['Class']} : ({x['Class_TotTrsClass']}, {x['Class_CountTrsPerClass']}, {x['Class_CountCardsPerClass']})", axis=1 )
node_positions = nodes_graph_df.apply(lambda x: f"{{'{x['Class']}' : '({x['Class_TotTrsClass']}, {x['Class_CountTrsPerClass']}, {x['Class_CountCardsPerClass']})'}}", axis=1 ).to_dict()
node_positions = [ast.literal_eval(y) for x,y in node_positions.items()]
node_positions = {k: ast.literal_eval(v) for x in node_positions for k, v in x.items()}

# Create a networkx graph
G = nx.Graph()
G.add_nodes_from(nodes)
G.add_weighted_edges_from(edges)

# Compute layout using a Plotly layout algorithm
pos = nx.spring_layout(G, dim=3)


# Create traces for nodes
node_trace = go.Scatter3d(
    # x=[pos[0] for pos in node_positions.values()],
    # y=[pos[1] for pos in node_positions.values()],
    # z=[pos[2] for pos in node_positions.values()],
    x=[pos[node][0] for node in nodes],
    y=[pos[node][1] for node in nodes],
    z=[pos[node][2] for node in nodes],
    mode='markers+text',
    # text=list(node_positions.keys()),
    text=nodes,
    textposition="bottom center",
    # hoverinfo=node_positions,
    marker=dict(
        size=10,
        color='blue'
    )
)

# Initialize an empty list to store annotations
annotations = []

# Create traces for edges
edge_x = []
edge_y = []
edge_z = []
for edge in edges:
    # x0, y0, z0 = node_positions[edge[0]]
    # x1, y1, z1 = node_positions[edge[1]]
    x0, y0, z0 = pos[edge[0]]
    x1, y1, z1 = pos[edge[1]]
    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])
    edge_z.extend([z0, z1, None])

    # Add annotation for the edge weight
    annotations.append(
        dict(
            x=(x0 + x1) / 2,
            y=(y0 + y1) / 2,
            z=(z0 + z1) / 2,
            text=str(edge[2]),
            showarrow=False
        )
    )

edge_trace = go.Scatter3d(
    x=edge_x,
    y=edge_y,
    z=edge_z,
    mode='lines',
    line=dict(color='black', width=1)
)


#### Parameter edges graph (Non-Baman == NB)
# graph df
selection = ['Class','Class_code','Class_TotTrsClass', 'Class_CountTrsPerClass', 'Class_CountCardsPerClass']
nodes_graph_df_NB = high_Card_DiscBetTotTrsOfAllAndOurTerms_df [selection]
nodes_graph_df_NB = nodes_graph_df_NB.drop_duplicates('Class')

# getting nodes and edges
column_TotClassesAreNotOurSet = high_Card_DiscBetTotTrsOfAllAndOurTerms_df.drop_duplicates('CardHash_')['Card_TotClassesAreNotOurSet']
column_TotClassesAreNotOurSet = column_TotClassesAreNotOurSet[pd.notna(column_TotClassesAreNotOurSet)]
list_TotClassesAreNotOurSet = list(column_TotClassesAreNotOurSet)
list_TotClassesAreNotOurSet = [ast.literal_eval(x) for x in list_TotClassesAreNotOurSet]
c_NB = Counter()
nodes_NB=set()
for row_set in list_TotClassesAreNotOurSet:
    row_list = list(row_set)
    nodes_NB.update(row_set)
    comb_NB = combinations(row_list, 2)
    c_NB.update(Counter(comb_NB))

edges_NB = [(k[0], k[1], v) for k,v in c_NB.most_common(40)]   ###----------->  esdges constraint
nodes_NB = list(nodes_NB)
nodes_NB = list(set([x for x in nodes_NB for y in edges_NB if x==y[0] or x == y[1]]))  ###----------->  nodes constraint
nodes_graph_df_NB['node_positions'] = nodes_graph_df_NB.apply(lambda x: f"{x['Class']} : ({x['Class_TotTrsClass']}, {x['Class_CountTrsPerClass']}, {x['Class_CountCardsPerClass']})", axis=1 )
node_positions_NB = nodes_graph_df_NB.apply(lambda x: f"{{'{x['Class']}' : '({x['Class_TotTrsClass']}, {x['Class_CountTrsPerClass']}, {x['Class_CountCardsPerClass']})'}}", axis=1 ).to_dict()
node_positions_NB = [ast.literal_eval(y) for x,y in node_positions_NB.items()]
node_positions_NB = {k: ast.literal_eval(v) for x in node_positions_NB for k, v in x.items()}

# Create a networkx graph
G_NB = nx.Graph()
G_NB.add_nodes_from(nodes_NB)
G_NB.add_weighted_edges_from(edges_NB)

# Compute layout using a Plotly layout algorithm
pos_NB = nx.spring_layout(G_NB, dim=3)


# Create traces for nodes
node_trace_NB = go.Scatter3d(
    # x=[pos[0] for pos in node_positions.values()],
    # y=[pos[1] for pos in node_positions.values()],
    # z=[pos[2] for pos in node_positions.values()],
    x=[pos_NB[node][0] for node in nodes_NB],
    y=[pos_NB[node][1] for node in nodes_NB],
    z=[pos_NB[node][2] for node in nodes_NB],
    mode='markers+text',
    # text=list(node_positions.keys()),
    text=nodes_NB,
    textposition="bottom center",
    # hoverinfo=node_positions,
    marker=dict(
        size=10,
        color='red'
    )
)

# Initialize an empty list to store annotations
annotations_NB = []

# Create traces for edges
edge_x_NB = []
edge_y_NB = []
edge_z_NB = []
for edge in edges_NB:
    # x0, y0, z0 = node_positions[edge[0]]
    # x1, y1, z1 = node_positions[edge[1]]
    x0, y0, z0 = pos_NB[edge[0]]
    x1, y1, z1 = pos_NB[edge[1]]
    edge_x_NB.extend([x0, x1, None])
    edge_y_NB.extend([y0, y1, None])
    edge_z_NB.extend([z0, z1, None])

    # Add annotation for the edge weight
    annotations_NB.append(
        dict(
            x=(x0 + x1) / 2,
            y=(y0 + y1) / 2,
            z=(z0 + z1) / 2,
            text=str(edge[2]),
            showarrow=False
        )
    )

edge_trace_NB = go.Scatter3d(
    x=edge_x_NB,
    y=edge_y_NB,
    z=edge_z_NB,
    mode='lines',
    line=dict(color='black', width=1)
)


#-------------------------------------------- visualisation

# Initialize Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

#----------------------------------------------------------------------------------- part 1

############## Section 1 --- Part 1

#### Clustering (Section 1 - Part 1)
@app.callback(
    Output("clustering", "figure"),
    Input("range-slider-clustering", "value"),
)
def update_chart(slider_range):
    low, high = slider_range
    mask = (high_Class_plot.Class_TotTrsClass > low) & (high_Class_plot.Class_TotTrsClass < high)

    fig = px.scatter_3d(
        high_Class_plot[mask],
        x="Class_TotTrsNotOurClass",
        y="Class_TotTrsOurClass",
        z="Class",
        # z="Card_CountTrsPerOurTerms",
        color="Class",
        # color="IsOurTerminal",
        # size="Card_DiscBetTotTrsOfAllAndOurTerms",
        # hover_data=["Card_TotTrsPerOurTerms"],
    )
    return fig

#### GEO (Section 1 - Part 1)
@app.callback(
    Output("graph1", "figure"),
    Input("type1", "value"),
)
def generate_chart(values):
    if values == "scatter_geo":
        fig = px.scatter_geo()
        fig.update_layout(mapbox_style="open-street-map")
        fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    else:
        fig = px.scatter_mapbox(
            high_Card_DiscBetTotTrsOfAllAndOurTerms_plot,
            lat="lat",
            lon="long",
            hover_data=["CardHash_","Card_DiscBetTotTrsOfAllAndOurTerms" ,"Card_TotTrsPerAllTerms_", "Card_TotTrsPerOurTerms_","Card_CountTrsPerAllTerms", "Card_CountTrsPerOurTerms",
                        "Card_TotTermsAreOurSet", "Card_TotTermsAreNotOurSet", "Card_TotClassesAreOurSet", "Card_TotClassesAreNotOurSet"],
            #             "Card_TotTermsAreNotOurSet","Card_TotTermsAreOurSet","Card_TotClassesAreNotOurSet","Card_TotClassesAreOurSet"],
            size="Card_DiscBetTotTrsOfAllAndOurTerms",
            color="Card_DiscBetTotTrsOfAllAndOurTerms",
            zoom=4,
        )
        fig.update_layout(mapbox_style="open-street-map")
    return fig

#### scatter plot (Section 1 - Part 1)
@app.callback(
    Output("scatter-plot-1", "figure"),
    Input("range-slider-1", "value"),
)
def update_chart_1(slider_range):
    low, high = slider_range
    mask = (high_Card_DiscBetTotTrsOfAllAndOurTerms_plot["Card_TotTrsPerAllTerms"] > low) & (high_Card_DiscBetTotTrsOfAllAndOurTerms_plot["Card_TotTrsPerAllTerms"] < high)
    fig = px.scatter(
        high_Card_DiscBetTotTrsOfAllAndOurTerms_plot[mask],
        x="Card_TotTrsPerAllTerms",
        y="Card_TotTrsPerOurTerms",
        color="Card_TotTrsPerOurTerms",
        size="Card_TotTrsPerAllTerms",
        hover_data=["CardHash_", "Card_TotTrsPerAllTerms_", "Card_TotTrsPerOurTerms_","Card_CountTrsPerAllTerms", "Card_CountTrsPerOurTerms",
                     "Card_TotTermsAreOurSet", "Card_TotTermsAreNotOurSet", "Card_TotClassesAreOurSet", "Card_TotClassesAreNotOurSet"],
    )
    return fig

#### BarChart (Section 1 - Part 1)

# bar chart slider
bar_slider = html.Div(
    [
        daq.GraduatedBar(
            id="bar",
            label="Filter minimum Discrepancy:",
            value=min_high_Card_DiscBetTotTrsOfAllAndOurTerms_barchart,
        ),
        html.Div(className='mb-5'),
        daq.Slider(
            id="slider",
            min=min_high_Card_DiscBetTotTrsOfAllAndOurTerms_barchart,
            max=max_high_Card_DiscBetTotTrsOfAllAndOurTerms_barchart,
            # value=(max_high_Card_DiscBetTotTrsOfAllAndOurTerms_barchart - min_high_Card_DiscBetTotTrsOfAllAndOurTerms_barchart)/20,
            value=min_high_Card_DiscBetTotTrsOfAllAndOurTerms_barchart,
        ),
    ]
)

@app.callback(
    Output("bar", "value"),
    Input("slider", "value"),
)
def update_output(value):
    return value

@app.callback(
    Output("graph_bar", "figure"),
    Input("slider", "value"),
)
def update_graph(value):
    dff = high_Card_DiscBetTotTrsOfAllAndOurTerms_barchart.loc[high_Card_DiscBetTotTrsOfAllAndOurTerms_barchart["Card_DiscBetTotTrsOfAllAndOurTerms"] >=  value]
    fig = px.bar(
        dff,
        x="CardHash_",
        y="tr",
        color="IsOurTerminal",
        hover_data=["Card_DiscBetTotTrsOfAllAndOurTerms_","tr_", "IsOurTerminal","CardHash_", "Card_TotTrsPerAllTerms_", "Card_TotTrsPerOurTerms_","Card_CountTrsPerAllTerms", "Card_CountTrsPerOurTerms",
                     "Card_TotTermsAreOurSet", "Card_TotTermsAreNotOurSet", "Card_TotClassesAreOurSet", "Card_TotClassesAreNotOurSet"],
        title="",
    )
    fig.update(layout=dict(title=dict(x=0.5)))
    return fig


#### second_BarChart (Section 1 - Part 1)

# second_BarChart slider2
bar_slider2 = html.Div(
    [
        daq.GraduatedBar(
            id="bar2",
            label="Filter minimum Discrepancy:",
            value=min_high_Card_DiscBetTotTrsOfAllAndOurTerms_barchart,
        ),
        html.Div(className='mb-5'),
        daq.Slider(
            id="slider2",
            min=min_high_Card_DiscBetTotTrsOfAllAndOurTerms_barchart,
            max=max_high_Card_DiscBetTotTrsOfAllAndOurTerms_barchart,
            # value=(max_high_Card_DiscBetTotTrsOfAllAndOurTerms_barchart - min_high_Card_DiscBetTotTrsOfAllAndOurTerms_barchart)/20,
            value=min_high_Card_DiscBetTotTrsOfAllAndOurTerms_barchart,
        ),
    ]
)


@app.callback(
    Output("bar2", "value"),
    Input("slider2", "value"),
)
def update_output(value):
    return value

@app.callback(
    Output("graph_bar2", "figure"),
    Input("slider2", "value"),
)
def update_graph(value):
    dff = high_Card_DiscBetTotTrsOfAllAndOurTerms_barchart.loc[high_Card_DiscBetTotTrsOfAllAndOurTerms_barchart["Card_DiscBetTotTrsOfAllAndOurTerms"] >=  value]
    fig = px.bar(
        dff,
        x="CardHash_",
        y="tr",
        color="Class",
        hover_data=["Card_DiscBetTotTrsOfAllAndOurTerms_","tr_", "IsOurTerminal","CardHash_", "Card_TotTrsPerAllTerms_", "Card_TotTrsPerOurTerms_","Card_CountTrsPerAllTerms", "Card_CountTrsPerOurTerms",
                     "Card_TotTermsAreOurSet", "Card_TotTermsAreNotOurSet", "Card_TotClassesAreOurSet", "Card_TotClassesAreNotOurSet"],
        title="",
    )
    fig.update(layout=dict(title=dict(x=0.5)))
    return fig


#### CSV (Section 1 - Part 1)
@app.callback(
    Output("clip1", "content"),
    Input("clip1", "n_clicks"),
    State("table1", "data"),
)
def custom_copy(_, data):
    dff = pd.DataFrame(data)
    return dff.to_csv(index=False) 

# Button Major_Discrepancies_Cards 1
@app.callback(
    Output("download_csv", "data"),
    Input("btn_csv", "n_clicks"),
    State("table1", "data"),
)
def download_csv(n_clicks, data):
    if not n_clicks:
        raise PreventUpdate
    df = high_Card_DiscBetTotTrsOfAllAndOurTerms_table
    csv_string = df.to_csv(index=False, encoding='utf-8')
    return  dict(content=csv_string, filename="Major_Discrepancies_Cards.csv")

#### Button Raw data (Section 1 - Part 1)
@app.callback(
    Output("download_csv_raw", "data"),
    Input("btn_csv_raw", "n_clicks"),
    prevent_initial_call=True
)
def download_csv_raw(n_clicks):
    if not n_clicks:
        raise PreventUpdate
    df = high_Card_DiscBetTotTrsOfAllAndOurTerms_df
    csv_string = df.to_csv(index=False, encoding='utf-8')
    return  dict(content=csv_string, filename="Raw_data.csv")


############## Section 2 --- Part 1 

#### GEO (Section 2 - Part 1)
@app.callback(
    Output("graph2", "figure"),
    Input("type2", "value"),
)
def generate_chart(values):
    if values == "scatter_geo":
        fig = px.scatter_geo()
        fig.update_layout(mapbox_style="open-street-map")
        fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    else:
        fig = px.scatter_mapbox(
            high_Term_plot,
            lat="lat",
            lon="long",
            hover_data=["TerminalId","Class" ,"Branch", "Term_TotTrsTerm_","Term_CountTrsPerTerm", "Term_CountCardsPerTerm", "Term_CountCardsPerTerm","IsOurTerminal"],
            size="Term_TotTrsTerm",
            color="IsOurTerminal",
            zoom=4,
        )
        fig.update_layout(mapbox_style="open-street-map")
    return fig

#### CSV (Section 2 - Part 1)
@app.callback(
    Output("clip2", "content"),
    Input("clip2", "n_clicks"),
    State("table2", "data"),
)
def custom_copy(_, data):
    dff = pd.DataFrame(data)
    return dff.to_csv(index=False) 

############## Section 3 --- Part 1

#### GEO (Section 3 - Part 1)
@app.callback(
    Output("graph3", "figure"),
    Input("type3", "value"),
)
def generate_chart(values):
    if values == "scatter_geo":
        fig = px.scatter_geo()
        fig.update_layout(mapbox_style="open-street-map")
        fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    else:
        fig = px.scatter_mapbox(
            high_Branch_plot,
            lat="lat",
            lon="long",
            hover_data=["Branch","Branch_TotTrsBranch_" ,"Branch_CountTrsPerBranch", "Branch_CountCardsPerBranch"],
            size="Branch_TotTrsBranch",
            color="Branch_CountTrsPerBranch",
            zoom=4,
        )
        fig.update_layout(mapbox_style="open-street-map")
    return fig

#### CSV (Section 3 - Part 1)
@app.callback(
    Output("clip3", "content"),
    Input("clip3", "n_clicks"),
    State("table3", "data"),
)
def custom_copy(_, data):
    dff = pd.DataFrame(data)
    return dff.to_csv(index=False)  


#----------------------------------------------------------------------------------- Part 2 --> app layout
# Define layout
app.layout = html.Div(
    [
############## Section 1 --- Part 2
        #### Clustering (Section 1 - Part 2)
        html.H4("Clustering"),
        dcc.Graph(id="clustering", style={'height':'200vh', 'width':'100%'}),
        html.P("Card_DiscBetTotTrsOfAllAndOurTerms:"),
        dcc.RangeSlider(
            id="range-slider-clustering",
            min=min_high_Class,
            max=max_high_Class,
            step=(max_high_Class - min_high_Class)/10,
            marks={min_high_Class: "min_high_Class", max_high_Class: "max_high_Class"},
            value=[min_high_Class, max_high_Class],
        ),
        
        #### GEO (Section 1 - Part 2)
        html.H4(" > GEO - Major Discrepancies among Cards --> High-Potential Cards needed actitivity to have transactions on Baman terminals!"),
        html.P(
            # "px.scatter_geo is used to plot points on globe across geolocations while "
            ""
        ),
        dcc.Graph(id="graph1", style={'height':'80vh', 'width':'100%'}),
        html.P(
            ""
        ),
        dcc.Dropdown(
            id="type1",
            # options=["scatter_mapbox", "scatter_geo"],
            options=["scatter_mapbox"],
            value="scatter_mapbox",
            clearable=False,
        ),

        #### Scatter-plot for DiscBetTotTrsOfAllAndOurTermsrepancies (Section 1 - Part 2)
        html.H4(" > Scatter Plot -  Major Discrepancies among Cards --> High-Potential Cards needed actitivity to have transactions on Baman terminals!"),
        html.H5("Size of Circles = Discrepancies Amount"),
        dcc.Graph(id="scatter-plot-1"),
        html.P("Filter by Card_TotTrsPerAllTerms width:"),
        dcc.RangeSlider(
            id="range-slider-1",
            min=min_Card_DiscBetTotTrsOfAllAndOurTerms,
            max=max_Card_DiscBetTotTrsOfAllAndOurTerms,
            step=(max_Card_DiscBetTotTrsOfAllAndOurTerms - min_Card_DiscBetTotTrsOfAllAndOurTerms) / 20,
            marks={min_Card_DiscBetTotTrsOfAllAndOurTerms: f"{min_Card_DiscBetTotTrsOfAllAndOurTerms}", max_Card_DiscBetTotTrsOfAllAndOurTerms: f"{max_Card_DiscBetTotTrsOfAllAndOurTerms}"},
            value=[min_Card_DiscBetTotTrsOfAllAndOurTerms, max_Card_DiscBetTotTrsOfAllAndOurTerms],
        ),

        ##### BarChart --> color = IsOurTerminal (Section 1 - Part 2)
        html.H4(" \n > Bar chart - Major Discrepancies among Cards --> High-Potential Cards needed actitivity to have transactions on Baman terminals"),
        dbc.Row(
            [
                dbc.Col(bar_slider, md=4),
                dbc.Col(dcc.Graph(id="graph_bar"), md=8, style={'height':'200%', 'width':'100%'}),
            ],
            align="center",
        ),

        #### BarChart --> color = Class (Section 1 - Part 2)
        html.H4(" \n > Bar chart - Major Discrepancies among Cards --> High-Potential Cards needed actitivity to have transactions on Baman terminals"),
        dbc.Row(
            [
                dbc.Col(bar_slider2, md=4),
                dbc.Col(dcc.Graph(id="graph_bar2"), md=8, style={'height':'200%', 'width':'100%'}),
            ],
            align="center",
        ),


        html.H4(
            "  > Major Discrepancies among Cards - Info CSV"
        ),

        #### Button Major_Discrepancies_Cards download (Section 1 - Part 2)
        html.Button("Download Major_Discrepancies_Cards CSV", id="btn_csv"),
        dcc.Download(id="download_csv"),
        ## end
        ## Button Raw data download 1
        html.Button("Download Raw Data CSV", id="btn_csv_raw"),
        dcc.Download(id="download_csv_raw"),
        ## end
        dcc.Clipboard(id="clip1", style={"fontSize": 20}),
        dash_table.DataTable(
        high_Card_DiscBetTotTrsOfAllAndOurTerms_table.to_dict("records"),
            [{"name": i, "id": i} for i in high_Card_DiscBetTotTrsOfAllAndOurTerms_table.columns],
            id="table1",
            page_size=10,
        ),


############## Section 2 --- Part 2
        #### GEO (Section 2 - Part 2)
        html.H4(f" > GEO - Highest Total Transactions Amounts at Terminals ({grand_selection} Terminals) --> Circle Size is Total Amount & Color is 'Yes or No' for 'Is Our Terminal?'"),
        html.P(
            # "px.scatter_geo is used to plot points on globe across geolocations while "
            ""
        ),
        dcc.Graph(id="graph2", style={'height':'80vh', 'width':'100%'}),
        html.P(
            ""
        ),
        dcc.Dropdown(
            id="type2",
            # options=["scatter_mapbox", "scatter_geo"],
            options=["scatter_mapbox"],
            value="scatter_mapbox",
            clearable=False,
        ),

        html.H4(
            f"  > Highest Total Transactions Amounts at Terminals ({grand_selection} Terminals) - Info CSV"
        ),
        dcc.Clipboard(id="clip2", style={"fontSize": 20}),
        dash_table.DataTable(
        high_Term_plot.to_dict("records"),
            [{"name": i, "id": i} for i in high_Term_plot.columns],
            id="table2",
            page_size=10,
        ),

############## Section 3 --- Part 2
        #### (Section 3 - Part 2)
        html.H4(" > GEO - Highest Total Transactions Amounts at Branches --> Circle Size is Total Amount & Color is Count of Transactions per each Branch"),
        html.P(
            # "px.scatter_geo is used to plot points on globe across geolocations while "
            ""
        ),
        dcc.Graph(id="graph3", style={'height':'80vh', 'width':'100%'}),
        html.P(
            ""
        ),
        dcc.Dropdown(
            id="type3",
            # options=["scatter_mapbox", "scatter_geo"],
            options=["scatter_mapbox"],
            value="scatter_mapbox",
            clearable=False,
        ),

        html.H4(
            "  > Highest Total Transactions Amounts at Branches - Info CSV"
        ),
        dcc.Clipboard(id="clip3", style={"fontSize": 20}),
        dash_table.DataTable(
        high_Branch_plot.to_dict("records"),
            [{"name": i, "id": i} for i in high_Branch_plot.columns],
            id="table3",
            page_size=10,
        ),

############## Section 4 --- Part 2
        #### edges graph (Section 4 - Part 2)
        dcc.Graph(
                id='edges_graph',
                style={'height':'200vh', 'width':'100%'},
                figure={
                    'data': [node_trace, edge_trace],
                    'layout': go.Layout(scene=dict(annotations=annotations,
                    xaxis=dict(showgrid=False, visible=False), 
                    yaxis=dict(showgrid=False, visible=False), 
                    zaxis=dict(showgrid=False, visible=False)))
                }
        ),
        dcc.Graph(
                id='edges_graph_NB',
                style={'height':'200vh', 'width':'100%'},
                figure={
                    'data': [node_trace_NB, edge_trace_NB],
                    'layout': go.Layout(scene=dict(annotations=annotations_NB,
                    xaxis=dict(showgrid=False, visible=False), 
                    yaxis=dict(showgrid=False, visible=False), 
                    zaxis=dict(showgrid=False, visible=False)))
                }
        )

    ]
)





# if __name__ == "__main__":
#     app.run_server(host= '0.0.0.0',debug=False,port=8050)

if __name__ == "__main__":
    app.run_server(debug=True)
