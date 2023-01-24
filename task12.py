import itertools
import random
import time
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

BLOOD_MATCH_TABLE = [
    [True, True, True, True],
    [False, True, False, True],
    [False, False, True, True],
    [False, False, False, True]
]


def blood_match_single_pair(sick, donor) -> bool:
    """
    check if the donor match the sick
    :param sick: blood type, 0=O, 1=A, 2=B, 3=AB
    :param donor: blood type, 0=O, 1=A, 2=B, 3=AB
    :return: true if match
    """
    return BLOOD_MATCH_TABLE[donor][sick]


def blood_match_two_pairs(sick1, donor1, sick2, donor2):
    """
    check if the two sick-donor pairs match for swap
    :param sick1: blood type, 0=O, 1=A, 2=B, 3=AB
    :param donor1: blood type, 0=O, 1=A, 2=B, 3=AB
    :param sick2: blood type, 0=O, 1=A, 2=B, 3=AB
    :param donor2: blood type, 0=O, 1=A, 2=B, 3=AB
    :return: true if the pairs match
    """
    return BLOOD_MATCH_TABLE[donor1][sick2] and BLOOD_MATCH_TABLE[donor2][sick1]


def generate_random_pairs_graph(N_pairs: int) -> nx.Graph:
    """
    generate random sick-donor pairs,
    and puts them in a graph where an edge is present between two pairs if they match for kidney swap
    :param N_pairs: number of random pairs to generate
    :return: graph representing the pairs and the matches between them
    """

    # generate random pairs
    pairs_list = [(i, random.randint(0, 3), random.randint(0, 3)) for i in range(N_pairs)]
    # filter out pairs where the sick and the donor match
    pairs_list = [(i, sick, donor) for i, sick, donor in pairs_list if not blood_match_single_pair(sick, donor)]

    # add pairs to the graph
    G = nx.Graph()
    G.add_nodes_from([i for i, _, _ in pairs_list])

    # iterate all pairs and check if they match
    for (i, sick1, donor1), (j, sick2, donor2) in itertools.combinations(pairs_list, 2):
        if blood_match_two_pairs(sick1, donor1, sick2, donor2):
            G.add_edge(i, j)

    return G


def calculate_kidney_chance(N_pairs: int, N_tries: int) -> float:
    """
    generate random sick donor pairs, and calculate the percentage of sicks that got a kidney.
    :param N_pairs: number of pairs to check each try
    :param N_tries: number of tries of thr random generation - more iterations reduces variance but takes more time
    :return: average fraction of sicks that got a kidney
    """
    matched = 0
    for _ in range(N_tries):
        G = generate_random_pairs_graph(N_pairs)

        # count pairs where the donor match the sick
        matched += N_pairs - G.number_of_nodes()

        # find maximum matching
        matching = nx.max_weight_matching(G)
        matched += len(matching)

    return (matched / N_tries) / N_pairs


if __name__ == '__main__':
    start_time = time.time()
    print(f'fraction of sicks that got a kidney: {calculate_kidney_chance(1000, 500)}')
    print(f'time: {time.time() - start_time}')
