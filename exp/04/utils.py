import pandas as pd
import math
import networkx as nx
import json
import numpy as np
import collections
from multiprocessing import Pool

def get_distance(pre_json, curr_json):
    """
    get distance between two decives
    """
    return {'x': curr_json['x'] - pre_json['x'],
            'y': curr_json['y'] - pre_json['y']}

def get_velocity_of_walker_id(walker_id, df_contacts):
    """
    get the approximated velocity of a walker.
    if the walker walks within ONE device's detection area, the velocity is not available
    """
    velocity = {}
    df = df_contacts.loc[(df_contacts['walker_id'] == walker_id)].sort_values('time')

    for (_,row1),(_,row2) in zip(df[:-1].iterrows(),df[1:].iterrows()):
        pre_time, pre_json = int(row1['time']), json.loads(row1['json'])
        curr_time, curr_json = int(row2['time']), json.loads(row2['json'])
        distance = get_distance(pre_json['agentPos'], curr_json['agentPos'])
        velocity[pre_time] = {'x': distance['x'] / (curr_time - pre_time),
                              'y': distance['y'] / (curr_time - pre_time)}
        
    return velocity

def get_position(df_agents):
    """
    get position (x, y) of each agent
    """
    id_to_pos, pos_to_id = {}, {}
    for _, row in df_agents.iterrows():
        jsondict = json.loads(row['json'])
        id_to_pos[jsondict['id']] = jsondict['position']
        pos_to_id[(jsondict['position']['x'], jsondict['position']['y'])] = jsondict['id']
        
    return id_to_pos, pos_to_id

def get_agent_next_prob(df_walkers, df_contacts):
    prob = collections.defaultdict(collections.Counter)
    walker_ids = df_walkers['id'].tolist()
    
    for walker_id in walker_ids:
        df = df_contacts.loc[(df_contacts['walker_id'] == walker_id)]
        times = set(df['time'].tolist())
        for time in times:
            row2 = df.loc[df['time'] == str(int(time) + 1)]
            if row2.shape[0] == 0:
                continue
                
            row1 = pd.DataFrame(df.loc[df['time'] == time]
            .sort_values(by='json',
                         key=lambda col: pd.Series([float(json.loads(c)['distance']) for c in col])))
#             row1 = df.iloc[df['json'].str.get('distance').astype(float).argsort()]
        
            
            agent_id1 = row1.iloc[0]['agent_id']
            agent_id2 = row2.iloc[0]['agent_id']
            prob[agent_id1][agent_id2] += 1
    return prob

def map_prob_to_pos(prob, id_to_pos):
    dic = {}
    for k, d in prob.items():
        tmp = {}
        for k2, v2 in d.items():
            tmp[(id_to_pos[k2]['x'], id_to_pos[k2]['y'])] = v2
        dic[(id_to_pos[k]['x'], id_to_pos[k]['y'])] = tmp
    return dic

def map_prob_to_agent_id(prob, pos_to_id):
    dic = {}
    for k, d in prob.items():
        tmp = {}
        if k not in pos_to_id:
            continue
        for k2, v2 in d.items():
            if k2 in pos_to_id:
                tmp[pos_to_id[k2]] = v2
        dic[pos_to_id[k]] = tmp
    return dic