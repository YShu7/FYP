import pandas as pd
import math
import networkx as nx
import json
import numpy as np
import collections
from multiprocessing import Pool
import statistics

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
    prob_dir = collections.defaultdict(collections.Counter)
    real_ids = set(df_walkers['real_id'].tolist())

    for real_id in real_ids:
        walker_ids = set(df_walkers.loc[df_walkers['real_id'] == real_id]['id'].tolist())
        df = df_contacts.loc[df_contacts['walker_id'].isin(walker_ids)]

        times = set(df['time'].tolist())
        for time in times:
            row2s = df.loc[(df['time'] == str(int(time) + 60))]
            if row2s.shape[0] == 0:
                continue

            row1s = df.loc[df['time'] == time]
#             .sort_values(by='json',
#                          key=lambda col: pd.Series([float(json.loads(c)['distance']) for c in col])))
    #             row1 = df.iloc[df['json'].str.get('distance').astype(float).argsort()]

            for _, row1 in row1s.iterrows():
                for _, row2 in row2s.iterrows():
                    agent_id1 = row1['agent_id']
                    agent_id2 = row2['agent_id']
                    json1 = json.loads(row1['json'])
                    json2 = json.loads(row2['json'])
                    x_dir = json2['agentPos']['x'] - json1['agentPos']['x']
                    y_dir = json2['agentPos']['y'] - json1['agentPos']['y']
        #             print("{:.2f}".format(x_dir))
                    prob[agent_id1][agent_id2] += 1
                    prob_dir["{:.2f}".format(x_dir)]["{:.2f}".format(y_dir)] += 1
    return prob, prob_dir

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

def get_mean_v(last_rows, pre_rows):
    velocities_x, velocities_y = [], []
    for _, row2 in last_rows.iterrows():
        json2 = json.loads(row2['json'])['agentPos']
        for _, row1 in pre_rows.iterrows():
            json1 = json.loads(row1['json'])['agentPos']
            velocities_x.append(json2['x'] - json1['x'])
            velocities_y.append(json2['y'] - json1['y'])
    vx, vy = statistics.mean(velocities_x), statistics.mean(velocities_y)
    return vx, vy

def get_direct_prob(prob_agent_id, agent_id1, agent_id2):
    try:
        return prob_agent_id[agent_id1][agent_id2] / sum(prob_agent_id[agent_id1])
    except:
        return 0
    
def get_dis_prob(json1, json2, vx, vy):
    dis = math.sqrt((float(json2['x']) - (float(json1['x']) + vx)) ** 2 + (float(json2['y']) - (float(json1['y']) + vy)) ** 2)
    if (dis > 50):
        return -float('inf')
    return 1 / dis if dis != 0 else 100

def get_direction_prob(json1, json2, prob_dir):
    try:
        x_dir, y_dir = "{:.2f}".format(json2['x'] - json1['x']), "{:.2f}".format(json2['y'] - json1['y'])
        return prob_dir[x_dir][y_dir] / sum(prob_dir[x_dir])
    except:
        return 0

def get_avg_prob(candidate_agent_probs, walker_id2, new_prob, weight):
    if walker_id2 in candidate_agent_probs:
        tmp = candidate_agent_probs[walker_id2][0] * candidate_agent_probs[walker_id2][1]
        return ((tmp + new_prob * weight) / (candidate_agent_probs[walker_id2][1] + weight), 
                candidate_agent_probs[walker_id2][1] + weight)
    else:
        return (new_prob, weight)