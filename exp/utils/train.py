import sqlite3
import pandas as pd

cnx = sqlite3.connect('../corona-sniffer/backend/data/database_100.db')


df_walks = pd.read_sql_query("SELECT * FROM walks JOIN walkers ON walkers.id = walks.walker_id", cnx)
df_walkers = pd.read_sql_query("SELECT * FROM walkers", cnx)
df_agents = pd.read_sql_query("SELECT * FROM agents", cnx)
df_contacts_tmp = pd.read_sql_query("SELECT * FROM contacts", cnx)
idx = df_contacts_tmp.groupby(['walker_id', 'time'])['distance'].transform(max) == df_contacts_tmp['distance']
df_contacts = df_contacts_tmp[idx]


def get_last_2_time(df, walker_id):
    time = sorted(set(df.loc[df['walker_id'] == walker_id]['time'].tolist()))
    if len(time) < 2:
        return None, None
    time1, time2 = time[-1], time[-2]
    return time1, time2

def train(walker_id1):
    posX, posY, negX, negY = [], [], [], []
    time1, time2 = get_last_2_time(df_contacts, walker_id1)
    if not time1 or not time2:
        return #continue
    
    start_time = time.time()
    # in case there're multiple agents tracking the walker
    last_rows = df_contacts.loc[(df_contacts['walker_id'] == walker_id1) & (df_contacts['time'] == time1)]
    pre_rows = df_contacts.loc[(df_contacts['walker_id'] == walker_id1) & (df_contacts['time'] == time2)]
    if last_rows.shape[0] == 0 or pre_rows.shape[0] == 0:
        return #continue
        
    if DEBUG: print('walker_id1: {}, # last_rows: {}, # pre_rows: {}'.format(walker_id1, last_rows.shape, pre_rows.shape))
        
    # among all the tracked records, consider mean velocity as real velocity
    vx, vy = utils.get_mean_v(last_rows, pre_rows)
    avg_dis = math.sqrt(vx ** 2 + vy ** 2)
    if DEBUG: print('velocity: ', time.time() - start_time)
    
    start_time = time.time()
    # get candidate points by time
    candidates = df_contacts.loc[(pd.to_numeric(df_contacts["time"]) < (int(time1) + 100)) & 
                                 (pd.to_numeric(df_contacts["time"]) > (int(time1))) & 
                                 (df_contacts['walker_id'] != walker_id1)]
    next_time = candidates['time'].min()
    candidates = candidates.loc[candidates['time'] == next_time]
    # idx = groupby(['walker_id'])['time'].transform(lambda x: pd.to_numeric(x).min()) == pd.to_numeric(candidates['time'])
    # candidates = candidates[idx]
    if DEBUG: print(f'{candidates.shape}: ', time.time() - start_time)
    
    if candidates.shape[0] == 0:
        return #continue
    
    candidate_agent_probs = {}
    for _, row1 in last_rows.iterrows():
        start_time = time.time()
        
        agent_id1, time1 = row1['agent_id'], row1['time']
        json1 = json.loads(row1['json'])['agentPos']
    
        for _, row2 in candidates.iterrows():
            res = ''
            agent_id2, walker_id2, time2 = row2['agent_id'], row2['walker_id'], row2['time']
            json2 = json.loads(row2['json'])['agentPos']
            
            isPos, thisX = True, []
            if df_walks.loc[df_walks['walker_id'] == walker_id1].iloc[0]['real_id'] == df_walks.loc[df_walks['walker_id'] == walker_id2].iloc[0]['real_id']:
                isPos = True
            else:
                isPos = False
            
            new_prob = 0
            
            # direct
            this_prob = utils.get_direct_prob(prob_agent_id, agent_id1, agent_id2)
            new_prob += this_prob
            res += 'direct: ' + str(this_prob)
            thisX.append(this_prob)

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
                    vx2, vy2 = utils.get_mean_v(last_rows2, pre_rows2)
            avg_dis2 = math.sqrt(vx2 ** 2 + vy2 ** 2)
            
            this_prob = utils.get_dis_prob(json1, json2, (vx+vx2)/2, (vy+vy2)/2, prob_move[agent_id1] if agent_id1 in prob_move else None,
                                           avg_dis=(avg_dis+avg_dis2)/2, time=int(row2['time']) - int(row1['time']))
            if this_prob <= 1e-10:
                continue
            new_prob += this_prob
            thisX.append(this_prob)
            res += ', distance: ' + str(this_prob)

            # direction
            this_prob = utils.get_direction_prob(json1, json2, prob_dir, agent_id1)
            thisX.append(this_prob)
            new_prob += this_prob
            res += ', directions: ' + str(this_prob)
                
            if DEBUG: print(res)
                
            if isPos:
                posX.append(thisX)
                posY.append(1)
            else:
                negX.append(thisX)
                negY.append(0)
        if DEBUG: print(f'candidates', time.time() - start_time)
        return (posX, posY, negX, negY)
#     if i % 100 == 99: print(f'{i+1}th/{len(walker_ids)} iter: {time.time() - start_time}')