Here you go — A* versions of both problems (Elevator Scheduling + Delivery Agent Routing), written in clean Python, exactly matching the type of algorithms given in Sivanandam & Deepa (AISC).

✅ 1. A* Algorithm for Elevator Queue Problem
State Representation

current_floor

pending_requests (tuple of floors left)

Goal State

All requests served → pending_requests == ()

Cost Function

Actual cost g(n) = total floors travelled so far

Heuristic h(n) = nearest pending request distance (admissible)

✔ A* Python Code — Elevator Scheduling
import heapq

def heuristic(current_floor, pending):
    if not pending:
        return 0
    return min(abs(current_floor - p) for p in pending)

def a_star_elevator(start_floor, requests):
    start_state = (start_floor, tuple(sorted(requests)))
    pq = []
    heapq.heappush(pq, (0, start_state, [start_floor], 0))   # (f, state, path, g)

    visited = set()

    while pq:
        f, (current, pending), path, g = heapq.heappop(pq)

        if (current, pending) in visited:
            continue
        visited.add((current, pending))

        # Goal test
        if not pending:
            return path

        for nxt in pending:
            cost = abs(current - nxt)
            new_pending = list(pending)
            new_pending.remove(nxt)
            new_state = (nxt, tuple(new_pending))

            g_new = g + cost
            h_new = heuristic(nxt, new_pending)
            f_new = g_new + h_new

            heapq.heappush(pq, (f_new, new_state, path + [nxt], g_new))

    return None


# Example usage
start = 5
requests = [2, 8, 3, 10, 7]

print("A* Elevator Order:", a_star_elevator(start, requests))

✔ Expected Output
A* Elevator Order: [5, 3, 2, 7, 8, 10]


(A* finds the same optimal path as greedy, but with search-based optimality guarantee.)

✅ 2. A* Algorithm for Delivery Agent Route (TSP-like problem)
State Representation

current_position

pending_locations to deliver to

path_travelled

Cost Function

Actual cost g(n) = total distance travelled

Heuristic h(n) = minimum of distances to remaining nodes
(admissible, optimistic)

✔ A* Python Code — Delivery Agent
import math, heapq

def dist(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def heuristic_delivery(current, pending):
    if not pending:
        return 0
    return min(dist(current, p) for p in pending)

def a_star_delivery(start, locations):
    start_state = (start, tuple(locations))
    pq = []
    heapq.heappush(pq, (0, start_state, [start], 0))

    visited = set()

    while pq:
        f, (current, pending), path, g = heapq.heappop(pq)

        if (current, pending) in visited:
            continue
        visited.add((current, pending))

        if not pending:
            return path

        for nxt in pending:
            d = dist(current, nxt)
            new_pending = list(pending)
            new_pending.remove(nxt)

            new_state = (nxt, tuple(new_pending))
            g_new = g + d
            h_new = heuristic_delivery(nxt, new_pending)
            f_new = g_new + h_new

            heapq.heappush(pq, (f_new, new_state, path + [nxt], g_new))

    return None


# Example
start = (0, 0)
locations = [(2, 3), (5, 4), (1, 1), (6, 1), (3, 2)]

print("A* Delivery Route:", a_star_delivery(start, locations))

✔ Expected Output

Likely output (A* optimal path):

A* Delivery Route: [(0, 0), (1, 1), (2, 3), (3, 2), (5, 4), (6, 1)]

🔥 Summary (Easy to Write in Journal)
Problem	Heuristic	A* Benefit
Elevator Scheduling	Nearest floor	Guarantees optimal travel path
Delivery Agent Routing	Nearest location	Guarantees minimum total route distance
Both are state-space search problems	h(n) = nearest pending element	Heuristic is admissible

If you want, I can also give
✅ Full algorithm explanation (Steps, diagrams)
✅ Pseudo-code (as required in AISc practical manuals)
✅ Expected output screenshots

Just tell me: "Give journal-ready explanation"