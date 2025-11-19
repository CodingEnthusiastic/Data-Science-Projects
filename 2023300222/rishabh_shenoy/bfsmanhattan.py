import heapq

goal = "123456780"

def manhattan(s):
    d = 0
    for i,ch in enumerate(s):
        if ch != '0':
            g = int(ch)-1
            d += abs(i//3 - g//3) + abs(i%3 - g%3)
    return d

moves = {
    0:[1,3], 1:[0,2,4], 2:[1,5],
    3:[0,4,6], 4:[1,3,5,7], 5:[2,4,8],
    6:[3,7], 7:[4,6,8], 8:[5,7]
}

def greedy_bfs(start):
    pq = []
    heapq.heappush(pq,(manhattan(start),start,""))
    seen = set()

    while pq:
        h,s,path = heapq.heappop(pq)
        if s == goal: return path

        if s in seen: continue
        seen.add(s)
        zi = s.index('0')

        for m in moves[zi]:
            lst = list(s)
            lst[zi], lst[m] = lst[m], lst[zi]
            ns = "".join(lst)
            if ns not in seen:
                heapq.heappush(pq,(manhattan(ns),ns,path+f"{s}->{ns}\n"))

start = "125340678"
print(greedy_bfs(start))
