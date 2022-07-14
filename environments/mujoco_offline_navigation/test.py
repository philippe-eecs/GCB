def bfs(start, goal):
    def add_edge(adj, src, dest):
    
        adj[src].append(dest)
        adj[dest].append(src)

    def BFS(adj, src, dest, v, pred, dist):
        queue = []
        visited = [False for i in range(v)]
        for i in range(v):
            dist[i] = 1000000
            pred[i] = -1
        visited[src] = True
        dist[src] = 0
        queue.append(src)
        while (len(queue) != 0):
            u = queue[0]
            queue.pop(0)
            for i in range(len(adj[u])):
                if (visited[adj[u][i]] == False):
                    visited[adj[u][i]] = True
                    dist[adj[u][i]] = dist[u] + 1
                    pred[adj[u][i]] = u
                    queue.append(adj[u][i])

                    if (adj[u][i] == dest):
                        return True
    
        return False

    def printShortestDistance(adj, s, dest, v):
        res = []

        pred=[0 for i in range(v)]
        dist=[0 for i in range(v)]
    
        if (BFS(adj, s, dest, v, pred, dist) == False):
            print("Given source and destination are not connected")

        path = []
        crawl = dest
        crawl = dest
        path.append(crawl)
        
        while (pred[crawl] != -1):
            path.append(pred[crawl])
            crawl = pred[crawl]
        # print("Shortest path length is : " + str(dist[dest]), end = '')
        # print("\nPath is : : ")
        for i in range(len(path)-1, -1, -1):
            res.append(v_to_points[path[i]])
            print(v_to_points[path[i]], end=' ')
        
        return res

    # def add_vertices():
    # v = 8


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

        # print(points_to_v[i])

    v = counter
    adj = [[] for i in range(v)]

    for i in range(1, len(D4RL_MAZE_LAYOUT) - 1):
        for j in range(1, len(D4RL_MAZE_LAYOUT[0]) - 1):

            if D4RL_MAZE_LAYOUT[i][j] != 1:
                currv = points_to_v[i][j]

                if D4RL_MAZE_LAYOUT[i+1][j] != 1:
                    add_edge(adj, currv, points_to_v[i+1][j])

                if D4RL_MAZE_LAYOUT[i][j+1] != 1:
                    add_edge(adj, currv, points_to_v[i][j+1])

    source = points_to_v[start[0]][start[1]]
    dest = points_to_v[int(goal[0])][int(goal[1])]

    return printShortestDistance(adj, source, dest, v)

