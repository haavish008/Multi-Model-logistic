
import unittest

import networkx as nx
import pandas as pd

from src.part18_topsis_integration import (
    calculate_topsis_ranking
)


class TestLogisticsSystem(unittest.TestCase):

    def setUp(self):

        self.graph = nx.Graph()

        self.graph.add_edge(
            "A",
            "B",
            length_m=1000,
            dynamic_cost=1.0
        )

        self.graph.add_edge(
            "B",
            "C",
            length_m=2000,
            dynamic_cost=2.0
        )

        self.graph.add_edge(
            "A",
            "C",
            length_m=5000,
            dynamic_cost=5.0
        )

    def test_graph_connectivity(self):

        self.assertTrue(
            nx.has_path(
                self.graph,
                "A",
                "C"
            )
        )

    def test_shortest_path(self):

        route = nx.astar_path(
            self.graph,
            "A",
            "C",
            weight="dynamic_cost"
        )

        self.assertEqual(
            route,
            ["A", "B", "C"]
        )

    def test_edge_costs_nonnegative(self):

        for u, v, attributes in self.graph.edges(data=True):

            self.assertGreaterEqual(
                attributes["dynamic_cost"],
                0
            )

    def test_route_edges_exist(self):

        route = ["A", "B", "C"]

        for u, v in zip(
            route[:-1],
            route[1:]
        ):

            self.assertTrue(
                self.graph.has_edge(u, v)
            )

    def test_blocked_edge(self):

        self.graph["A"]["B"]["dynamic_cost"] = float("inf")

        route = nx.astar_path(
            self.graph,
            "A",
            "C",
            weight="dynamic_cost"
        )

        self.assertEqual(
            route,
            ["A", "C"]
        )

    def test_topsis_ranking(self):

        alternatives = pd.DataFrame({
            "algorithm": ["Route A", "Route B"],
            "risk": [0.2, 0.5],
            "time": [5.0, 8.0],
            "cost": [100.0, 150.0],
            "distance": [100.0, 150.0],
            "carbon": [10.0, 20.0]
        })

        result = calculate_topsis_ranking(
            alternatives
        )

        self.assertEqual(
            result.iloc[0]["algorithm"],
            "Route A"
        )

        self.assertEqual(
            result.iloc[0]["topsis_rank"],
            1
        )


if __name__ == "__main__":
    unittest.main()
