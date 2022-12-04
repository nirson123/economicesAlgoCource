import networkx as nx
from math import log, inf
from functools import reduce
from heapq import heapify, heappop, heappush


def cycle_with_multiplication_less_then_1(graph: nx.DiGraph, weight_attribute: str) -> list:
    """
    find a cycle in graph, where the multiplication of the edges in the cycle is < 1.
    if such a cycle doesn't exists, raise a NetworkXException
    :param graph: Digraph from networkX
    :param weight_attribute: the edge attribute where the edges' weights are stored
    :return: the cycle, as a list of nodes
    """
    work_graph = graph.copy()
    # log all the weights, to convert from multiplication (< 1) to addition (< 0)
    log_weights = {(u, v): log(d[weight_attribute]) for u, v, d in graph.edges(data=True)}
    nx.set_edge_attributes(work_graph, log_weights, 'log_weight')

    # find cycle and return it
    return nx.find_negative_cycle(work_graph, min(list(work_graph.nodes())), 'log_weight')


def test_cycle_with_multiplication_less_then_1():
    """
    tests for cycle_with_multiplication_less_then_1
    :return:
    """
    def test_less_then_1(cycle, graph: nx.DiGraph):
        """
        make sure the cycle has multiplication < 1
        :param cycle:
        :param graph:
        :return:
        """
        mul = reduce(lambda x, y: x*y, [graph.get_edge_data(u, v)['weight'] for u, v in zip(cycle, cycle[1:])], 1)
        assert mul < 1

    #  1 - (1) -> 2
    #  |  \(0.1)
    # (3)    \
    #  v        \(2)
    #  3 - (2) -> 4
    # cycle: [1, 4, 4] or [1, 3, 4, 1]
    g1 = nx.DiGraph({1: [2, 3, 4], 2: [], 3: [4], 4: [1]})
    nx.set_edge_attributes(g1, {(1, 2): 1, (1, 3): 3, (1, 4): 2, (3, 4): 2, (4, 1): 0.1}, 'weight')
    test_less_then_1(cycle_with_multiplication_less_then_1(g1, 'weight'), g1)

    #  1 - (1) -> 2
    #  |  \(0.1)
    # (3)    \
    #  v        \
    #  3 - (2) -> 4
    # cycle:[1, 3, 4, 1]
    g1.remove_edge(1, 4)
    test_less_then_1(cycle_with_multiplication_less_then_1(g1, 'weight'), g1)

    #  1 - (1) -> 2
    #  | \(0.1)
    # (3)   \
    #  |       \
    #  v         \(2)
    #  3          4
    # no cycles
    g1.remove_edge(3, 4)
    try:
        cycle_with_multiplication_less_then_1(g1, 'weight')
        assert False
    except nx.NetworkXException:
        pass

    # 1-> 2 -> 3 -> 4 -> 1
    # cycle has multiplication > 1
    g2 = nx.DiGraph({1: [2], 2: [3], 3: [4], 4: [1]})
    nx.set_edge_attributes(g2, {(1, 2): 2, (2, 3): 0.5, (3, 4): 2, (4, 1): 0.8}, 'weight')
    try:
        cycle_with_multiplication_less_then_1(g2, 'weight')
        assert False
    except nx.NetworkXException:
        pass


def cycle_with_multiplication_less_then_1_dijkstra(graph: nx.DiGraph, weight_attribute: str):
    """
    find a cycle in graph, where the multiplication of the edges in the cycle is < 1.
    if such a cycle doesn't exists, return None.
    The search use dijkstra algorithm, with edges weights multiplication instead of sum.
    :param graph: Digraph from networkX
    :param weight_attribute: the edge attribute where the edges' weights are stored
    :return: the cycle, as a list of nodes
    """

    # initialize all nodes data
    nx.set_node_attributes(graph, False, 'visited')
    nx.set_node_attributes(graph, inf, 'path_mul_weight')
    nx.set_node_attributes(graph, None, 'father')

    # pick arbitrary starting node
    starting_node = list(graph.nodes())[0]
    graph.nodes[starting_node]['visited'] = True
    graph.nodes[starting_node]['path_mul_weight'] = 1

    q = list(graph.nodes())

    while q:

        # this is inefficient  - linear search every time. can e replaced by min heap.
        # this is not a heap only because python heaps are annoying
        # to work with, and I didn't feel like implementing my own
        node = min(q, key=lambda n: graph.nodes[n]['path_mul_weight'])
        q.remove(node)
        graph.nodes[node]['visited'] = True

        for nei in graph.neighbors(node):
            # new node
            if not graph.nodes[nei]['visited']:
                d = graph.nodes[node]['path_mul_weight'] * graph.edges[node, nei][weight_attribute]
                if d < graph.nodes[nei]['path_mul_weight']:     # set new "distance"
                    graph.nodes[nei]['path_mul_weight'] = d
                    graph.nodes[nei]['father'] = node
            # found a (potential ) cycle
            elif graph.nodes[node]['path_mul_weight'] * graph.edges[node, nei][weight_attribute] < 1:
                cycle = [node, nei]
                true_cycle = True
                while not cycle[0] == nei:
                    # not a cycle, just re-visiting an old node from a different starting point
                    if not graph.nodes[cycle[0]]['father']:
                        true_cycle = False
                        break
                    cycle.insert(0, graph.nodes[cycle[0]]['father'])
                if true_cycle:  # found a (good) cycle, return it
                    return cycle

    return None     # no cycle found


def test_cycle_with_multiplication_less_then_1_dijkstra():
    """
    tests for cycle_with_multiplication_less_then_1_dijkstra
    :return:
    """
    def test_less_then_1(cycle, graph: nx.DiGraph):
        """
        make sure the cycle has multiplication < 1
        :param cycle:
        :param graph:
        :return:
        """
        mul = reduce(lambda x, y: x*y, [graph.get_edge_data(u, v)['weight'] for u, v in zip(cycle, cycle[1:])], 1)
        assert mul < 1

    #  1 - (1) -> 2
    #  |  \(0.1)
    # (3)    \
    #  v        \(2)
    #  3 - (2) -> 4
    # cycle: [1, 4, 4] or [1, 3, 4, 1]
    g1 = nx.DiGraph({1: [2, 3, 4], 2: [], 3: [4], 4: [1]})
    nx.set_edge_attributes(g1, {(1,2): 1, (1,3): 3, (1,4): 2, (3,4): 2, (4,1): 0.1}, 'weight')
    test_less_then_1(cycle_with_multiplication_less_then_1_dijkstra(g1, 'weight'), g1)

    #  1 - (1) -> 2
    #  |  \(0.1)
    # (3)    \
    #  v        \
    #  3 - (2) -> 4
    # cycle:[1, 3, 4, 1]
    g1.remove_edge(1, 4)
    test_less_then_1(cycle_with_multiplication_less_then_1_dijkstra(g1, 'weight'), g1)


    #  1 - (1) -> 2
    #  | \(0.1)
    # (3)   \
    #  |       \
    #  v         \(2)
    #  3          4
    # no cycles
    g1.remove_edge(3, 4)
    assert cycle_with_multiplication_less_then_1_dijkstra(g1, 'weight') is None

    # 1-> 2 -> 3 -> 4 -> 1
    # cycle has multiplication > 1
    g2 = nx.DiGraph({1: [2], 2: [3], 3: [4], 4: [1]})
    nx.set_edge_attributes(g2, {(1, 2): 2, (2, 3): 0.5, (3, 4): 2, (4, 1): 0.8}, 'weight')
    assert cycle_with_multiplication_less_then_1_dijkstra(g2, 'weight') is None


def main():
    test_cycle_with_multiplication_less_then_1()
    test_cycle_with_multiplication_less_then_1_dijkstra()


if __name__ == '__main__':
    main()

