from collections import deque
from itertools import count

from mujoco_offline_navigation.test import bfs

import numpy as np

def get_waypoints(maze_arena, spawn, goal):
    waypoints = []

    start = (spawn, )
    start = maze_arena.world_to_grid_positions(start)
    start = np.asarray(start, dtype=np.int32)[0][:2]

    goal = (goal, )
    goal = maze_arena.world_to_grid_positions(goal)
    goal = np.asarray(goal, dtype=np.int32)[0][:2]

    print("spawn is", start)
    print("goal is", goal)

    wps = bfs(start, goal)
    wps = wps[:len(wps) - 1]
    wps.append(goal)
    # goal = np.asarray(goal, dtype=np.int32)[0][:2]

    # maze = maze_arena.maze.entity_layer

    # stack = deque()
    # stack.append((start[0], start[1]))
    # visited = {(start[0], start[1])}
    # previous = {(start[0], start[1]): None}
    # while True:
    #     top = stack.pop()
    #     if top[0] == goal[0] and top[1] == goal[1]:
    #         break

    #     for di, dj in zip([1, -1, 0, 0], [0, 0, 1, -1]):
    #         new_i = top[0] + di
    #         new_j = top[1] + dj
    #         if new_i >= 0 and new_i < maze.shape[
    #                 1] and new_j >= 0 and new_j < maze.shape[0] and maze[
    #                     new_i, new_j] != '*' and not (new_i, new_j) in visited:
    #             stack.append((new_i, new_j))
    #             visited.add((new_i, new_j))
    #             previous[(new_i, new_j)] = (top[0], top[1])

    # last = top
    # while last is not None:
    #     waypoints.append(last)
    #     last = previous[last]

    # waypoints = list(reversed(waypoints))
    return maze_arena.grid_to_world_positions(wps)


def add_vertices(maze):
    v = 8
    adj = [[] for i in range(v)]

    D4RL_MAZE_LAYOUT = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                        [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
                        [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                        [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
                        [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]

    v_to_points = {}
    points_to_v = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]

    counter = 0

    for i in range(1, len(points_to_v)):
        for j in range(1, len(points_to_v[0])):
            if points_to_v[i][j] != 1:
                points_to_v[i][j] = counter
                v_to_points[counter] = (i, j)
                counter += 1
    
    v = counter

    for i in range(1, len(D4RL_MAZE_LAYOUT) - 1):
        for j in range(1, len(D4RL_MAZE_LAYOUT[0]) - 1):
            currv = points_to_v[i][j]

            if D4RL_MAZE_LAYOUT[i+1][j] != 1:
                add_edge(adj, currv, points_to_v[i+1][j])

            if D4RL_MAZE_LAYOUT[i][j+1] != 1:
                add_edge(adj, currv, points_to_v[i][j+1])

    source = points_to_v[1][1]
    dest = points_to_v[1][9]

    printShortestDistance(adj, source, dest, v)