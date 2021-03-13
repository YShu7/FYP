import networkx as nx
import pandas as pd
import collections
import math
from scipy.optimize import linear_sum_assignment

DEBUG = True

def get_info(df_contacts):
    walker_ids = set(df_contacts['walker_id'].tolist())
    id_to_time = {}
    id_to_agent = {}
    start_time_to_id = collections.defaultdict(list)
    end_time_to_id = collections.defaultdict(list)
    for walker_id in walker_ids:
        candidates = df_contacts.loc[df_contacts['walker_id'] == walker_id].sort_values('time')
        min_time, max_time = candidates.iloc[0]['time'], candidates.iloc[-1]['time']
        id_to_time[walker_id] = (int(min_time), int(max_time))
        id_to_agent[walker_id] = (candidates.iloc[0]['agent_id'], candidates.iloc[-1]['agent_id'])
        start_time_to_id[int(min_time)].append(walker_id)
        end_time_to_id[int(max_time)].append(walker_id)
    return id_to_time, id_to_agent, start_time_to_id, end_time_to_id

def get_link_list_based_time(df_contacts, INTERVAL):
    id_to_time, id_to_agent, start_time_to_id, end_time_to_id = get_info(df_contacts)
    link_list = {}

    times = sorted(list(end_time_to_id.keys()))
    for i, time in enumerate(times):
        batch_walker_ids, candidates = get_candidates(df_contacts, time, start_time_to_id, end_time_to_id, INTERVAL)
        if batch_walker_ids is None or candidates is None:
            continue
        if len(batch_walker_ids) == 1 and candidates.shape[0] == 1:
            link_list[batch_walker_ids[0]] = candidates.iloc[0]['walker_id']

    return link_list

def link(link_list, df_walks):
    """
    Connect each linked path into a single path
    """
    graph = nx.Graph([(i, j) for i, j in link_list.items() if i and j])
    connected_components = sorted(nx.connected_components(graph), key=len, reverse=True)
    print(len(connected_components))
    print(connected_components)

    df_walks_connected = pd.DataFrame(df_walks)
    for i, component in enumerate(connected_components):
        new_id = i
        for c in component:
            df_walks_connected = df_walks_connected.replace({'walker_id': {c: str(new_id) + 'res'}})

    return df_walks_connected

def get_candidates(df_contacts, time, start_time_to_id, end_time_to_id, INTERVAL):
    batch_walker_ids = end_time_to_id[time]
    if len(batch_walker_ids) == 0:
        return None, None

    candidate_ids = start_time_to_id[time + INTERVAL]
    next_times = [time + INTERVAL]
    candidates = df_contacts.loc[
        (df_contacts['walker_id'].isin(candidate_ids)) & (df_contacts['time'].isin(next_times))]

    if len(candidate_ids) < len(batch_walker_ids):
        # candidate_ids += start_time_to_id[time + 2 * INTERVAL]
        # next_times.append(time + 2 * INTERVAL)
        # candidates = df_contacts.loc[
        #     (df_contacts['walker_id'].isin(candidate_ids)) & (df_contacts['time'].isin(next_times))]
        #
        # if len(candidate_ids) < len(batch_walker_ids):
        #     candidate_ids += start_time_to_id[time + 3 * INTERVAL]
        #     next_times.append(time + 3 * INTERVAL)
        #     candidates = df_contacts.loc[
        #         (df_contacts['walker_id'].isin(candidate_ids)) & (df_contacts['time'].isin(next_times))]

        if candidates.shape[0] == 0:
            return None, None
    return batch_walker_ids, candidates

def get_weight_matrix(batch_walker_ids_prob):
    walker_id_to_index, walker_index_to_id = {}, []
    candidate_id_to_index, candidate_index_to_id = {}, []

    for k, values in batch_walker_ids_prob.items():
        walker_id_to_index[k] = len(walker_id_to_index)
        walker_index_to_id.append(k)
        for c, v in values.items():
            if c not in candidate_id_to_index:
                candidate_id_to_index[c] = len(candidate_id_to_index)
                candidate_index_to_id.append(c)
    matrix = [[1e-32] * len(candidate_id_to_index) for _ in range(len(walker_id_to_index))]

    for k, values in batch_walker_ids_prob.items():
        for c, v in values.items():
            matrix[walker_id_to_index[k]][candidate_id_to_index[c]] = -math.log(v)
    return matrix, candidate_index_to_id, walker_index_to_id

def get_link_list_based_model(df_contacts, INTERVAL, to_ix, model, N, M):
    """
    Map agent id to its most possible next agent.
    """
    id_to_time, id_to_agent, start_time_to_id, end_time_to_id = get_info(df_contacts)

    link_list, reversed_link_list = {}, {}
    no_candidates = 0

    visited_walker_ids = set()
    times = sorted(list(end_time_to_id.keys()))
    for i, time in enumerate(times):
        batch_walker_ids, candidates = get_candidates(df_contacts, time, start_time_to_id, end_time_to_id, INTERVAL)

        if batch_walker_ids is None or candidates is None:
            no_candidates += 1
            continue
        if i % 500 == 499:
            print(f'{i + 1}...')

        batch_walker_ids_prob = {}
        for walker_id2 in batch_walker_ids:
            if walker_id2 not in visited_walker_ids:
                walker_id2s = [walker_id2]
                while walker_id2 in reversed_link_list:
                    walker_id2 = reversed_link_list[walker_id2]
                    walker_id2s.append(walker_id2)
                walk2 = df_contacts.loc[df_contacts['walker_id'].isin(walker_id2s)].sort_values('time')

                batch_walker_ids_prob[walk2.iloc[-1]['walker_id']] = model.compute_proba(walk2, candidates, to_ix,
                                                                                         N, M, INTERVAL)

        matrix, candidate_index_to_id, walker_index_to_id = get_weight_matrix(batch_walker_ids_prob)
        row_ind, col_ind = linear_sum_assignment(matrix)

        for row_idx, col_idx in zip(row_ind, col_ind):
            if matrix[row_idx][col_idx] >= -math.log(1e-3):
                continue
            link_list[walker_index_to_id[row_idx]] = candidate_index_to_id[col_idx]
            reversed_link_list[candidate_index_to_id[col_idx]] = walker_index_to_id[row_idx]
            visited_walker_ids.add(walker_index_to_id[row_idx])

    return link_list