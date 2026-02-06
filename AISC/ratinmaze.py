N = 4

maze = [
    [1,0,0,0],
    [1,1,0,1],
    [0,1,0,0],
    [1,1,1,1]
]

sol = [[0]*N for _ in range(N)]

def safe(x,y):
    return 0 <= x < N and 0 <= y < N and maze[x][y] == 1

def solve(x,y):
    if x == N-1 and y == N-1:
        sol[x][y] = 1
        return True

    if safe(x,y):
        sol[x][y] = 1

        if solve(x+1,y): return True
        if solve(x,y+1): return True

        sol[x][y] = 0
    return False

if solve(0,0):
    for r in sol: print(r)
else:
    print("No path")
