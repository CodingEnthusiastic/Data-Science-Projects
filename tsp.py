import math

def dist(a,b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def tsp_nn(points):
    n = len(points)
    visited = [False]*n
    path = [0]
    visited[0] = True
    curr = 0

    for _ in range(n-1):
        nxt = -1
        best = float('inf')
        for i in range(n):
            if not visited[i]:
                d = dist(points[curr],points[i])
                if d < best:
                    best = d
                    nxt = i
        visited[nxt] = True
        path.append(nxt)
        curr = nxt

    path.append(0)
    return path

points = [(0,0),(1,5),(5,1),(4,7),(8,3)]
print(tsp_nn(points))
