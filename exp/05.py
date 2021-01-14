import sqlite3
import pandas as pd
import collections
import networkx as nx
import json

cnx = sqlite3.connect('../corona-sniffer/backend/data/database_3000.db')
df_agents = pd.read_sql_query("SELECT * FROM agents", cnx)
df_walker_contacts = pd.read_sql_query("SELECT * FROM walkers INNER JOIN contacts", cnx)


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

def trans_to_list(df_walker_contacts):
    real_ids = set(df_walker_contacts['real_id'].tolist())
    agent_id_tracks = []
    for real_id in real_ids:
        agent_id_track = df_walker_contacts.loc[df_walker_contacts['real_id'] == real_id].sort_values('time')['agent_id'].tolist()
        agent_id_tracks.append(agent_id_track)
    return agent_id_tracks

def get_agent_next_prob(agent_id_tracks):
    aa_counts = collections.defaultdict(lambda: collections.Counter()) #collections.defaultdict(lambda: collections.defaultdict(lambda: collections.Counter()))
    agent_counts = collections.Counter()

    for track in agent_id_tracks:
        pre_agent = track[0]
        agent_counts[track[0]] += 1
        for agent in track[1:]:
            aa_counts[pre_agent][agent] += 1
            agent_counts[agent] += 1
            pre_agent = agent

    V = sum(agent_counts.values())
    for prev_agent in agent_counts.keys():
        T = len(aa_counts[prev_agent].keys())
        Z = V - T
        agent_counts[prev_agent] += T
        for agent in agent_counts.keys():
            if agent not in aa_counts[prev_agent]:
                aa_counts[prev_agent][agent] = T / Z

    return aa_counts, agent_counts

id_to_pos, pos_to_id = get_position(df_agents)

aa_counts, agent_counts = get_agent_next_prob(trans_to_list(df_walker_contacts))
file = open('model.txt', 'w')
file.write(json.dumps({
    'aa_counts': aa_counts,
    'agent_counts': agent_counts
}, indent=2, sort_keys=True))
file.close()

cnx = sqlite3.connect('../corona-sniffer/backend/data/database_3000_test.db')

df_walks = pd.read_sql_query("SELECT * FROM walks JOIN walkers ON walkers.id = walks.walker_id", cnx)
df_walkers = pd.read_sql_query("SELECT * FROM walkers", cnx)
df_contacts = pd.read_sql_query("SELECT * FROM contacts", cnx)
df_agents = pd.read_sql_query("SELECT * FROM agents", cnx)

def algo(df_contacts, aa_counts, agent_counts):
    df_contacts = df_contacts.sort_values('time')
    N = len(set(df_contacts['agent_id'].tolist()))

    prev_time, prev_agent = None, None
    visited = set()
    groups = []

    for i, row in df_contacts.iterrows():
        if prev_agent == row['agent_id']:
            continue
        if row['walker_id'] in visited:
            prev_agent = row['agent_id']
            continue

        df_considered = df_contacts.iloc[i:]
        # agents = set(df_considered['agent_id'].tolist())

        N = df_considered.shape[0]
        prob = [[None] * N for _ in range(N)]
        bp = [[None] * N for _ in range(N)]

        for t in range(1, 10):
            for s, row in df_considered.iterrows():
                time, tag, walker_id = row['time'], row['agent_id'], row['walker_id']
                curr_max = -float('inf')
                curr_max_idx = 0

                for prev_s, prev_row in df_considered.iterrows():
                    prev_time, prev_tag, prev_walker_id = row['time'], row['agent_id'], row['walker_id']
                    tmp = prob[t - 1][prev_s] * (aa_counts[prev_tag][tag] / agent_counts[prev_tag])
                    if tmp > curr_max:
                        curr_max = tmp
                        curr_max_idx = prev_s
                prob[t][s] = curr_max
                bp[t][s] = curr_max_idx

        prev_bp = curr_max_idx
        group = set()
        for i, bp in enumerate(reversed(bp)):
            idx = T - i - 1
            group.add(df_considered.iloc[prev_bp]['walker_id'])
            prev_bp = bp[prev_bp]
            visited.add(df_considered.iloc[prev_bp]['walker_id'])
        groups.append(group)

        # agent_max_prob = max(prob[prev_agent].items(), key=operator.itemgetter(1))[0]
        # prev_time, prev_agent = row['time'], row['agent_id']

    return groups

groups = algo(df_contacts, aa_counts, agent_counts)

df_walks_connected = pd.DataFrame(df_walks)
for i, component in enumerate(groups):
    new_id = i
    for c in component:
        df_walks_connected = df_walks_connected.replace(c, new_id)

real_ids = set(df_walks_connected['real_id'].tolist())
correct, total, tried = 0, 0, 0
for real_id in real_ids:
    id_list = df_walks_connected.loc[df_walks_connected['real_id'] == real_id].sort_values('time')['walker_id'].tolist()
    old_id_list = df_walks.loc[df_walks['real_id'] == real_id].sort_values('time')['walker_id'].tolist()

    tried += sum([0 if id == old_id else 1 for (id, old_id) in zip(id_list, old_id_list)])
    
    dic = collections.Counter(id_list)
    correct += max(dic.values())
    total += len(id_list)

cnx.execute('DROP TABLE IF EXISTS walks_attached')
df_walks_connected.to_sql('walks_attached', con=cnx)

print('correct/tried: {}/{} {}%'.format(correct, tried, correct/tried*100))
print('correct/total: {}/{} {}%'.format(correct, total, correct/total*100))
print('baseline: {}/{} {}%'.format(len(real_ids) * 2, total, len(real_ids) * 2 / total * 100))

