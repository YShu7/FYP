import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def draw(connected_components, x_coordinates, y_coordinates, T, axis):
    fig, axs = plt.subplots(1, T, figsize=(3*T, 3))
    for i in range(T):
        colors = matplotlib.cm.rainbow(np.linspace(0, 1, len(connected_components[i])))
        for c, component in enumerate(connected_components[i]):
            for j in component:
                axs[i].axis(axis)
                axs[i].scatter(x_coordinates[i][j], y_coordinates[i][j], color=colors[c])
    plt.show()

def draw_predicated_points(connected_components, fixed_points_arr, random_points_arr, T, axis):
    fig, axs = plt.subplots(1, T, figsize=(3 * T, 3))
    for i in range(T):
        colors = matplotlib.cm.rainbow(np.linspace(0, 1, len(connected_components[i])))
        for c, component in enumerate(connected_components[i]):
            for j in component:
                axs[i].axis(axis)
                if j in fixed_points_arr[i]:
                    axs[i].scatter(fixed_points_arr[i][j][0], fixed_points_arr[i][j][1], color=colors[c])
                elif j in random_points_arr[i] and random_points_arr[i][j] is not None:
                    for rand in random_points_arr[i][j]:
                        axs[i].scatter(rand[0], rand[1], facecolors='none', edgecolor=colors[c])
    plt.show()