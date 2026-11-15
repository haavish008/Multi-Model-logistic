
import networkx as nx
import pytest

from src.maritime.part25_dstar_replanning import DStarLite

def create_test_graph():
    graph = nx.DiGraph()

    graph.add_edge("A", "B", weight=5)
    graph.add_edge("B", "D", weight=5)
    graph.add_edge("A", "C", weight=3)
    graph.add_edge("C", "D", weight=8)

    return graph

def test_astar_shortest_path():
    graph = create_test_graph()

    route = nx.astar_path(
        graph,
        "A",
        "D",
        heuristic=lambda a, b: 0,
        weight="weight"
    )

    assert route == ["A", "B", "D"]

def test_astar_route_cost():
    graph = create_test_graph()

    route = nx.astar_path(
        graph,
        "A",
        "D",
        heuristic=lambda a, b: 0,
        weight="weight"
    )

    cost = nx.path_weight(graph, route, weight="weight")

    assert cost == 10

def test_dstar_lite_initial_route():
    graph = create_test_graph()

    planner = DStarLite(graph, "A", "D")
    planner.compute_shortest_path()

    route = planner.get_path()

    assert route == ["A", "B", "D"]

def test_dstar_lite_after_edge_closure():
    graph = create_test_graph()

    planner = DStarLite(graph, "A", "D")
    planner.compute_shortest_path()

    planner.update_edge("B", "D", float("inf"))
    planner.compute_shortest_path()

    route = planner.get_path()

    assert route == ["A", "C", "D"]

def test_unreachable_destination():
    graph = nx.DiGraph()
    graph.add_edge("A", "B", weight=2)

    planner = DStarLite(graph, "A", "B")
    planner.update_edge("A", "B", float("inf"))
    planner.compute_shortest_path()

    assert planner.get_path() == []
