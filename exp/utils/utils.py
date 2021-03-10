import pandas as pd
import math
import networkx as nx
import json
import numpy as np
import collections
from multiprocessing import Pool
import statistics
import sqlite3
import time as Time

IMPOSSIBLE = 1e-10

def get_last_2_time(walk, walker_id):
    time = sorted(set(walk['time'].tolist()))
    if len(time) < 2:
        return None, None
    time1, time2 = time[-1], time[-2]
    return time1, time2

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

def get_agent_next_prob(df_walkers, df_contacts, agent_ids):
    prob = collections.defaultdict(collections.Counter)
    prob_dir = collections.defaultdict(collections.Counter)
    prob_move = {}
    real_ids = set(df_walkers['real_id'].tolist())

    for real_id in real_ids:
        start_time = Time.time()
        walker_ids = set(df_walkers.loc[df_walkers['real_id'] == real_id]['id'].tolist())
        df = df_contacts.loc[df_contacts['walker_id'].isin(walker_ids)]
#         print(Time.time() - start_time)
        
        start_time = Time.time()
        times = set(df['time'].tolist())
        for time in times:
            row2s = df.loc[(df['time'] == int(time) + 60)]
            if row2s.shape[0] == 0:
                continue

            row1s = df.loc[df['time'] == time]

            for _, row1 in row1s.iterrows():
                for _, row2 in row2s.iterrows():
                    agent_id1 = row1['agent_id']
                    agent_id2 = row2['agent_id']
                    json1 = json.loads(row1['json'])
                    json2 = json.loads(row2['json'])
                    x_dir = json2['agentPos']['x'] - json1['agentPos']['x']
                    y_dir = json2['agentPos']['y'] - json1['agentPos']['y']
                    prob[agent_id1][agent_id2] += 1
                    prob_dir["{:.2f}".format(x_dir)]["{:.2f}".format(y_dir)] += 1

    for agent_id1 in agent_ids:
        for agent_id2 in agent_ids:
            if agent_id1 in prob:
                if agent_id2 in prob[agent_id1]:
                    pass
                else:
                    prob[agent_id1][agent_id2] += 0.1
            else:
                prob[agent_id1][agent_id2] += 0.1

    for k in prob_dir.keys():
        prob_move[k] = 1 - prob_dir[k][k] / sum(prob_dir[k].values())
    return prob, prob_dir, prob_move

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
        time2 = int(row2['time'])
        json2 = json.loads(row2['json'])['agentPos']
        for _, row1 in pre_rows.iterrows():
            time1 = int(row1['time'])
            json1 = json.loads(row1['json'])['agentPos']
            velocities_x.append((json2['x'] - json1['x']) / (time2 - time1))
            velocities_y.append((json2['y'] - json1['y']) / (time2 - time1))
    vx, vy = statistics.mean(velocities_x), statistics.mean(velocities_y)
    return vx, vy

def get_direct_prob(prob_agent_id, agent_id1, agent_id2):
    try:
        return prob_agent_id[agent_id1][agent_id2] / sum(prob_agent_id[agent_id1].values())
    except:
        return 0
    
def get_dis_prob(json1, json2, vx, vy, prob_move, avg_dis=50, time=60, THRESHOLD=0.7):
    p = prob_move
    
    dis = math.sqrt((float(json2['x']) - (float(json1['x']) + vx * time)) ** 2 + (float(json2['y']) - (float(json1['y']) + vy * time)) ** 2)        
    dis_static = math.sqrt((float(json2['x']) - (float(json1['x']))) ** 2 + (float(json2['y']) - (float(json1['y']))) ** 2)
    
    if (dis > avg_dis * time) and (dis_static > avg_dis * time):
        return IMPOSSIBLE
    
    if p is None:
        return max(1 / dis if dis != 0 else 100, 1 / dis_static if dis_static != 0 else 100)
        
    if p < THRESHOLD:
        return 1 / dis_static if dis_static != 0 else 100
    
    return 1 / dis if dis != 0 else 100

def get_direction_prob(json1, json2, prob_dir, agent_id):
    try:
        x_dir, y_dir = "{:.2f}".format(json2['x'] - json1['x']), "{:.2f}".format(json2['y'] - json1['y'])
        return prob_dir[agent_id][x_dir][y_dir] / sum(prob_dir[agent_id][x_dir].values())
    except:
        return 0

def get_avg_prob(candidate_agent_probs, walker_id2, new_prob, weight):
    if walker_id2 in candidate_agent_probs:
        tmp = candidate_agent_probs[walker_id2][0] * (1 / candidate_agent_probs[walker_id2][1])
        return ((tmp + new_prob * (1 / weight)) * (candidate_agent_probs[walker_id2][1] + weight), 
                candidate_agent_probs[walker_id2][1] + weight)
    else:
        return (new_prob, weight)
    
def get_prob(df_contacts, prob_agent_id, agent_id1, agent_id2, walker_id2, json1, json2, prob_dir, vx, vy, prob_move, time):
    prob = []
    
    # direct
    this_prob = get_direct_prob(prob_agent_id, agent_id1, agent_id2)
    prob.append(this_prob)
    
    # distance
    timec1, timec2 = get_last_2_time(df_contacts, walker_id2)
    if not timec1 or not timec2:
        vx2, vy2 = vx, vy
    else:
        last_rows2 = df_contacts.loc[(df_contacts['walker_id'] == walker_id2) & (df_contacts['time'] == timec1)]
        pre_rows2 = df_contacts.loc[(df_contacts['walker_id'] == walker_id2) & (df_contacts['time'] == timec2)]
        if last_rows2.shape[0] == 0 or pre_rows2.shape[0] == 0:
            vx2, vy2 = vx, vy
        else:
            vx2, vy2 = get_mean_v(last_rows2, pre_rows2)
    avg_dis2 = math.sqrt(((vx+vx2)/2) ** 2 + ((vy+vy2)/2) ** 2)

    this_prob = get_dis_prob(json1, json2, (vx+vx2)/2, (vy+vy2)/2, 
                             prob_move[agent_id1] if agent_id1 in prob_move else None,
                             avg_dis=avg_dis2, time=time)

    prob.append(this_prob)
    
    # direction
    this_prob = get_direction_prob(json1, json2, prob_dir, agent_id1)
    prob.append(this_prob)

    return prob