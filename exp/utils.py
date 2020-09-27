import random
import numpy as np
import networkx as nx
import pandas as pd
import functools
import collections
from sympy import *

def get_coordinates(T, N, R, step):
    random.seed(2)
    x_coordinates = np.zeros((T, N))
    x_coordinates[0] = [random.uniform(0, R) for _ in range(N)]
    y_coordinates = np.zeros((T, N))
    y_coordinates[0] = [random.uniform(0, R) for _ in range(N)]
    for t in range(T - 1):
        x_steps = [1 if random.random() > 0.5 else 0 for _ in range(N)]
        y_steps = [1 if random.random() > 0.5 else 0 for _ in range(N)]
        x_coordinates[t + 1] = x_coordinates[t] + np.multiply(x_steps, step)
        y_coordinates[t + 1] = y_coordinates[t] + np.multiply(y_steps, step)
    return x_coordinates, y_coordinates

def get_distances(x_coordinates, y_coordinates):
    T, N = x_coordinates.shape
    distances = np.zeros((T, N, N))
    for t in range(T):
        x, y = x_coordinates[t], y_coordinates[t]
        for i, (x1, y1) in enumerate(zip(x, y)):
            for j, (x2, y2) in enumerate(zip(x, y)):
                distances[t, i, j] = np.linalg.norm([x1-x2, y1-y2])
    return distances

def get_contacted_pairs(threshold, distances):
    T, N, _ = distances.shape
    contacted_pairs = [[] for i in range(T)]
    for t in range(T):
        for i in range(N):
            for j in range(i+1, N):
                if distances[t][i][j] <= threshold:
                    contacted_pairs[t].append((i, j, distances[t][i][j]))
    return contacted_pairs

def get_clusters(contacted_pairs):
    graphs = [nx.Graph([(i, j) for i, j, dis in p]) for p in contacted_pairs]
    connected_components = [sorted(nx.connected_components(g), key=len, reverse=True) for g in graphs]
    return connected_components

def get_triangles(contacted_pairs):
    graphs = [nx.Graph([(i, j) for i, j, dis in p]) for p in contacted_pairs]
    triangles = [[x for x in nx.enumerate_all_cliques(g) if len(x) == 3] for g in graphs]
    return triangles

def get_degrees(a, b, c, tag):
    queue = [[b, c, a], [a, c, b], [a, b, c]]
    res = []
    for ac, ab, bc in queue:
        cosine = (ac ** 2 + ab ** 2 - bc ** 2) / (2 * ac * ab)
        res.append(np.arccos(max(-1, min(cosine, 1))))
        if cosine < -1 or cosine > 1:
            print(tag, cosine, a, b, c)
    return res

def retrieve_distance(df, i, j):
    arr = df.loc[(df['i'] == i) & (df['j'] == j)]['distance'].values.tolist()
    arr += df.loc[(df['i'] == j) & (df['j'] == i)]['distance'].values.tolist()
    return arr[0]

def get_extra_info(idx_count, b, c, x_coordinates, y_coordinates, t):
    if idx_count[b] > idx_count[c]:
        return b, (x_coordinates[t][b], y_coordinates[t][b])
    else:
        return c, (x_coordinates[t][c], y_coordinates[t][c])

def predicate_points(T, triangles, contacted_pairs, R, x_coordinates, y_coordinates):
    fixed_points_arr, random_points_arr = [], []
    T, N = x_coordinates.shape
    for i in range(T):
        df = pd.DataFrame(contacted_pairs[i], columns=['i', 'j', 'distance'])
        fixed_points, random_points = {}, {}

        def cmp(tri_a, tri_b):
            count_a = sum([1 if p in fixed_points else 0 for p in tri_a])
            count_b = sum([1 if p in fixed_points else 0 for p in tri_b])
            if count_a - count_b == 0:
                return sum([1 if p in random_points else 0 for p in tri_a]) - sum([1 if p in random_points else 0 for p in tri_b])
            else:
                return count_a - count_b

        idx_count = collections.Counter()
        for a, b, c in triangles[i]:
            idx_count[a] += 1
            idx_count[b] += 1
            idx_count[c] += 1

        sorted_triangles = sorted(triangles[i], key=functools.cmp_to_key(cmp))
        known_points = []
        # for a, b, c in triangles[i]:
        while len(sorted_triangles) != 0:
            a, b, c = sorted_triangles.pop()
            if a in fixed_points and b in fixed_points and c in fixed_points:
                continue
            # have 2 fixed points
            elif (a in fixed_points and b in fixed_points) or (a in fixed_points and c in fixed_points) or (b in fixed_points and c in fixed_points):
                pass
            # have 1 fixed point
            elif a in fixed_points:
                x, idx = get_extra_info(idx_count, b, c, x_coordinates, y_coordinates, i)
                fixed_points[x] = idx
                known_points.append(x)
            elif b in fixed_points:
                x, idx = get_extra_info(idx_count, a, c, x_coordinates, y_coordinates, i)
                fixed_points[x] = idx
                known_points.append(x)
            elif c in fixed_points:
                x, idx = get_extra_info(idx_count, a, b, x_coordinates, y_coordinates, i)
                fixed_points[x] = idx
                known_points.append(x)
            # have 0 fixed point
            else:
                x, idx = get_extra_info(idx_count, a, b, x_coordinates, y_coordinates, i)
                fixed_points[x] = idx
                known_points.append(x)
                if x == a:
                    a, b = b, a
                x, idx = get_extra_info(idx_count, a, c, x_coordinates, y_coordinates, i)
                fixed_points[x] = idx
                known_points.append(x)

            if a in fixed_points:
                if b not in fixed_points:
                    a, b = b, a
                else:
                    a, c = c, a
            x1, y1 = fixed_points[b]
            x2, y2 = fixed_points[c]
            bc = retrieve_distance(df, b, c)
            ac = retrieve_distance(df, a, c)
            ab = retrieve_distance(df, a, b)
            degree, _, _ = get_degrees(abs(y2-y1), bc, abs(x2-x1), (b, c, np.linalg.norm([x1-x2, y1-y2]))) #np.arccos(max(-1, min(np.dot([1, 0], [(x2 - x1) / bc, (y2 - y1) / bc]), 1)))
            degree_a, degree_b, degree_c = get_degrees(bc, ac, ab, 2)
            # x, y = Symbol('x'), Symbol('y')
            # solved_value = solve([(y2 - y) ** 2 + (x2 - x) ** 2 - ac ** 2, (y1 - y) ** 2 + (x1 - x) ** 2 - ab ** 2], [x, y])
            # print(solved_value)
            # if len(solved_value) == 0:
            #     break
            # (x3, y3), (x4, y4) = solved_value[:2]
            tmp = []
            factors = [[1, 1], [1, -1], [-1, 1], [-1, -1]]
            # if y1 < y2:
            for fx, fy in factors:
                x, y = fx * np.cos(degree - degree_b) * ab + x1, fy * np.sin(degree - degree_b) * ab + y1
                if (abs(np.linalg.norm([x2-x, y2-y]) - ac) < 1e-5):
                    tmp.append((x, y))
            for fx, fy in factors:
                x, y = fx * np.cos(degree + degree_b) * ab + x1, fy * np.sin(degree + degree_b) * ab + y1
                if (abs(np.linalg.norm([x2 - x, y2 - y]) - ac) < 1e-5):
                    tmp.append((x, y))
            # else:
            for fx, fy in factors:
                x, y = fx * np.cos(degree - degree_c) * ac + x2, fy * np.sin(degree - degree_c) * ac + y2
                if (abs(np.linalg.norm([x2 - x, y2 - y]) - ab) < 1e-5):
                    tmp.append((x, y))
            for fx, fy in factors:
                x, y = fx * np.cos(degree + degree_c) * ac + x2, fy * np.sin(degree + degree_c) * ac + y2
                if (abs(np.linalg.norm([x2 - x, y2 - y]) - ab) < 1e-5):
                    tmp.append((x, y))

            if a in random_points and random_points[a] is not None:
                fixed = False
                for x, y in random_points[a]:
                    for xt, yt in tmp:
                        if x == xt and y == yt:
                            fixed_points[a] = (x, y)
                            fixed = True
                            random_points[a] = None
                            break
                if not fixed:
                    random_points[a] += tmp
            else:
                random_points[a] = tmp
            sorted_triangles = sorted(sorted_triangles, key=functools.cmp_to_key(cmp))
        print(f'Known points: {len(known_points)} {known_points}')
        # print(random_points)
        # for k, v in random_points.items():
        #     if k in fixed_points:
        #         continue
        #     fixed_points[k] = v[0]
        fixed_points_arr.append(fixed_points)
        random_points_arr.append(random_points)
    return fixed_points_arr, random_points_arr

def evaluate(t, x_coordinates, y_coordinates, fixed_points_arr, random_points_arr, N):
    count, rand_count = 0, 0
    distance = 0

    for i, (x, y) in enumerate(zip(x_coordinates[t], y_coordinates[t])):
        if i in fixed_points_arr[t]:
            count += 1
            distance += np.linalg.norm([x - fixed_points_arr[t][i][0], y - fixed_points_arr[t][i][1]])
        elif i in random_points_arr[t]:
            rand_count += 1

    print(f'Matched: {count}/{N}, Random: {rand_count}/{N}, Average error distance: {distance / count if count != 0 else None}')

def random_guess(d, b):
    random.seed(2)
    random_degree = random.uniform(0, 180)
    return (b[0] + np.cos(random_degree) * d, b[1] + np.sin(random_degree) * d)