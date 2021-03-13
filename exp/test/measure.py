import collections
import numpy as np
import seaborn as sns
from scipy.optimize import linear_sum_assignment

DEBUG = False


def draw(matrix, row_ind, col_ind):
    sns.set_theme()
    sns.set(rc={'figure.figsize': (11.7, 11.7)})

    assign_matrix = matrix
    for i, j in zip(row_ind, col_ind):
        assign_matrix[i][j] = -matrix[i][j]

    ax = sns.heatmap(assign_matrix)


def accuracy(df_walks_connected):
    """
    Accuracy: How much percentage of the grouped ids are really from one group
    """
    correct, total = 0, 0
    cluster_edge, cluster_p, accpeted_cluster = 0, 0, 0
    empty_cluster = 0
    points = 0

    walker_ids = list(set(df_walks_connected['walker_id'].tolist()))
    ids = list(set(df_walks_connected['id'].tolist()))
    real_ids = list(set(df_walks_connected['real_id'].tolist()))

    real_path_length = {}
    for real_id in real_ids:
        real_path_length[real_id] = len(set(df_walks_connected.loc[df_walks_connected['real_id'] == real_id]['id']))
        total += len(set(df_walks_connected.loc[df_walks_connected['real_id'] == real_id]['walker_id'])) - 1

    accepted_histogram = [0] * 11

    for i, walker_id in enumerate(walker_ids):
        walk = df_walks_connected.loc[df_walks_connected['walker_id'] == walker_id].sort_values('walk_time')

        walk_real_id, walk_id = [], []
        for j, row in walk.iterrows():
            if len(walk_id) == 0 or row['id'] != walk_id[-1]:
                walk_id.append(row['id'])
                walk_real_id.append(row['real_id'])

        sub_path_id, sub_path_length, tmp = [], [], 0
        for pre, curr in zip(walk_real_id[:-1], walk_real_id[1:]):
            if pre == curr:
                correct += 1
                tmp += 1
            else:
                sub_path_id.append(pre)
                sub_path_length.append(tmp + 1)
                tmp = 0
            total += 1
        sub_path_id.append(walk_real_id[-1])
        sub_path_length.append(tmp + 1)

        # print(sub_path_id, sub_path_length)

        rate = (max(sub_path_length)-1) / (real_path_length[sub_path_id[np.argmax(sub_path_length)]]-1)
        accepted_histogram.append(rate)
        if max(sub_path_length) / (real_path_length[sub_path_id[np.argmax(sub_path_length)]])>= 0.6:
            accpeted_cluster += 1
        if max(sub_path_length)-1 == 0:
            empty_cluster += 1

        cluster_edge += max(sub_path_length) - 1
        cluster_p += max(sub_path_length)
        points += max(collections.Counter(walk_real_id).values())

    accept_rate = accpeted_cluster/len(walker_ids) * 100
    acc = correct / total * 100
    pe = cluster_edge / (len(ids)-len(real_ids)) * 100
    pp = cluster_p / (len(ids)) * 100
    print(f'Accept Rate: {accpeted_cluster}/{len(walker_ids)} {accept_rate}')
    print(f'Accept Rate No Empty: {accpeted_cluster}/{len(walker_ids)-empty_cluster} {accpeted_cluster/(len(walker_ids)-empty_cluster)}')
    print(f'Accuracy: {correct}/{total} {acc}%')
    print(f'Purity E: {cluster_edge}/{len(ids)-len(walker_ids)} {pe}')
    print(f'Purity P: {cluster_p}/{len(ids)} {pp}')
    print(f'Purity P - no linkage: {points}/{len(ids)} {pp}')
    return accept_rate, acc, pe, pp, accepted_histogram

def purity(matrix):
    """
    Purity: How much percentage of one group is correctly grouped together (only consider the largest sub-group of grouped ids)
    """

    correct, total = 0, 0
    for j in range(len(matrix[0])):
        col = matrix[:, j]
        l = max(col)

        correct += l - 1
        total += sum(col) - 1

    print(f'Purity: {correct}/{total} {correct / total * 100}%')


def measure(df_walks_connected):
    walker_ids = list(set(df_walks_connected['walker_id'].tolist()))
    real_ids = list(set(df_walks_connected['real_id'].tolist()))

    matrix = [[0] * len(real_ids) for _ in range(len(walker_ids))]

    if DEBUG: print(len(walker_ids))

    for i, walker_id in enumerate(walker_ids):
        walk = df_walks_connected.loc[df_walks_connected['walker_id'] == walker_id]
        dic = collections.defaultdict(list)
        for _, row in walk.iterrows():
            dic[row['real_id']].append(row['id'])

        for j, real_id in enumerate(real_ids):
            matrix[i][j] = -len(set(dic[real_id])) if real_id in dic else 0

    matrix = np.array(matrix)
    row_ind, col_ind = linear_sum_assignment(matrix)
    print(len(walker_ids), len(real_ids), len(row_ind), len(col_ind))

    correct = -matrix[row_ind, col_ind].sum()
    total = len(set(df_walks_connected['id'].tolist()))

    # walker_acc = 0
    # for i, r in enumerate(row_ind):
    #     walker_acc += (matrix[r][col_ind[i]]-1) / (sum(matrix[r])-1)
    #
    # real_acc = 0
    # accept_num = 0
    # for i, c in enumerate(col_ind):
    #     rate = (matrix[row_ind[i]][c]-1) / (sum([matrix[j][c] for j in range(len(walker_ids))])-1)
    #     real_acc += rate
    #     accept_num += 1 if rate > 0.6 else 0
    #
    # print(f'correct/total: {correct}/{total} {correct / total * 100}%')
    # print(f'accuracy: {walker_acc / len(walker_ids) * 100}')
    # print(f'purity: {real_acc / len(real_ids) * 100}')
    # print(f'accept rate: {accept_num / len(real_ids) * 100}')

    matrix = -matrix

    walker_acc, walker_all = 0, 0
    for i, r in enumerate(row_ind):
        walker_acc += matrix[r][col_ind[i]] - 1
        walker_all += sum(matrix[r]) - 1

    real_acc, real_all = 0, 0
    accept_num = 0
    for i, c in enumerate(col_ind):
        rate = (matrix[row_ind[i]][c] - 1) / (sum([matrix[j][c] for j in range(len(walker_ids))]) - 1)
        real_all += (sum([matrix[j][c] for j in range(len(walker_ids))]) - 1)
        real_acc += (matrix[row_ind[i]][c] - 1)
        accept_num += 1 if rate > 0.6 else 0

    correctness = correct / total * 100
    accuracy_mapped = walker_acc / walker_all * 100
    purity_mapped = real_acc / real_all * 100
    rate = accept_num / len(real_ids) * 100
    print(f'correct/total: {correct}/{total} {correctness}%')
    print(f'accuracy: {accuracy_mapped}')
    print(f'precision: {purity_mapped}')
    print(f'accept rate: {rate}')

    print('\n')

    accept_rate, acc, pe, pp, accepted_histogram = accuracy(df_walks_connected)
    # purity(df_walks_connected)

    # draw(matrix, row_ind, col_ind)
    return correctness, accuracy_mapped, purity_mapped, rate, \
           accept_rate, acc, pe, pp, accepted_histogram
