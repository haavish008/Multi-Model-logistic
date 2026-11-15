import heapq
import itertools
import random
import math
import numpy as np
import pandas as pd
import networkx as nx


def edge_cost(graph, u, v):
    """Return the non-negative routing cost of an edge."""
    data = graph[u][v]
    cost = data.get("dynamic_cost", data.get("weight", data.get("length_m", 1.0)))
    return max(0.0, float(cost))


def euclidean_heuristic(a, b):
    """Coordinate-based heuristic for tuple coordinate nodes."""
    try:
        return math.hypot(float(a[0]) - float(b[0]),
                          float(a[1]) - float(b[1]))
    except (TypeError, ValueError, IndexError):
        return 0.0


class DStarLite:

    def __init__(self, graph, start, goal):
        self.graph = graph
        self.start = start
        self.last_start = start
        self.goal = goal

        self.g = {}
        self.rhs = {goal: 0.0}
        self.km = 0.0

        self.queue = []
        self.counter = itertools.count()

        self.insert(goal)

    def value(self, table, node):
        return table.get(node, float("inf"))

    def heuristic(self, a, b):
        return euclidean_heuristic(a, b)

    def calculate_key(self, node):
        minimum = min(
            self.value(self.g, node),
            self.value(self.rhs, node)
        )

        return (
            minimum + self.heuristic(self.start, node) + self.km,
            minimum
        )

    def insert(self, node):
        key = self.calculate_key(node)

        heapq.heappush(
            self.queue,
            (key[0], key[1], next(self.counter), node)
        )

    def top_key(self):

        while self.queue:
            k1, k2, _, node = self.queue[0]

            if (k1, k2) == self.calculate_key(node):
                return k1, k2

            heapq.heappop(self.queue)

        return float("inf"), float("inf")

    def pop(self):

        while self.queue:
            k1, k2, _, node = heapq.heappop(self.queue)

            if (k1, k2) == self.calculate_key(node):
                return (k1, k2), node

        return (float("inf"), float("inf")), None

    def update_vertex(self, node):

        if node != self.goal:

            self.rhs[node] = min(
                (
                    edge_cost(self.graph, node, neighbor)
                    + self.value(self.g, neighbor)
                    for neighbor in self.graph.neighbors(node)
                ),
                default=float("inf")
            )

        if self.value(self.g, node) != self.value(self.rhs, node):
            self.insert(node)

    def compute_shortest_path(self):

        iterations = 0
        maximum_iterations = max(
            10000,
            self.graph.number_of_nodes() * 100
        )

        while (
            self.top_key() < self.calculate_key(self.start)
            or self.value(self.rhs, self.start)
            != self.value(self.g, self.start)
        ):

            iterations += 1

            if iterations > maximum_iterations:
                raise RuntimeError("D* Lite iteration limit exceeded.")

            old_key, node = self.pop()

            if node is None:
                break

            new_key = self.calculate_key(node)

            if old_key < new_key:
                self.insert(node)

            elif self.value(self.g, node) > self.value(self.rhs, node):

                self.g[node] = self.value(self.rhs, node)

                for predecessor in self.graph.neighbors(node):
                    self.update_vertex(predecessor)

            else:

                self.g[node] = float("inf")

                self.update_vertex(node)

                for predecessor in self.graph.neighbors(node):
                    self.update_vertex(predecessor)

    def extract_path(self):

        self.compute_shortest_path()

        if not np.isfinite(self.value(self.g, self.start)):
            raise nx.NetworkXNoPath("D* Lite found no route.")

        current = self.start
        path = [current]
        visited = {current}

        for _ in range(self.graph.number_of_nodes()):

            if current == self.goal:
                return path

            candidates = []

            for neighbor in self.graph.neighbors(current):

                score = (
                    edge_cost(self.graph, current, neighbor)
                    + self.value(self.g, neighbor)
                )

                candidates.append((score, neighbor))

            if not candidates:
                break

            score, next_node = min(candidates, key=lambda x: x[0])

            if not np.isfinite(score):
                break

            if next_node in visited:
                raise RuntimeError("D* Lite generated a cyclic route.")

            path.append(next_node)
            visited.add(next_node)
            current = next_node

        raise nx.NetworkXNoPath("Unable to extract a complete D* Lite route.")

    def update_edge(self, u, v, new_cost):

        if not self.graph.has_edge(u, v):
            raise ValueError("The requested edge does not exist.")

        if new_cost < 0:
            raise ValueError("Edge cost cannot be negative.")

        self.graph[u][v]["dynamic_cost"] = float(new_cost)

        self.update_vertex(u)
        self.update_vertex(v)

    def block_road(self, u, v):
        """Set an existing edge to a very high cost."""
        self.update_edge(u, v, 1e15)

    def move_start(self, new_start):

        self.km += self.heuristic(
            self.last_start,
            new_start
        )

        self.last_start = new_start
        self.start = new_start
