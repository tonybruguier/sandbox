import itertools

import numpy as np
import networkx
import scipy.optimize

import cirq


def brute_force(graph, n):
    bitstrings = np.array(list(itertools.product(range(2), repeat=n)))
    mat = networkx.adjacency_matrix(graph, nodelist=sorted(graph.nodes))
    vecs = (-1) ** bitstrings
    vals = 0.5 * np.sum(vecs * (mat @ vecs.T).T, axis=-1)
    vals = 0.5 * (graph.size() - vals)
    return np.max(vals)


def main():
    # Set problem parameters
    n = 6
    # Generate a random 3-regular graph on n nodes
    graph = networkx.random_regular_graph(3, n)

    # Compute best possible cut value via brute force search
    max_cut = brute_force(graph, n)

    # Make qubits
    circuit = cirq.Circuit()
    qubits = cirq.LineQubit.range(n)

    # Prepare uniform superposition
    circuit.append(cirq.H.on_each(*qubits))

    theta = np.pi / 2.0
    for i, j in graph.edges:
        circuit.append(cirq.ZZPowGate(exponent=(2.0 * theta / np.pi), global_shift=-0.5).on(qubits[i], qubits[j]))



if __name__ == '__main__':
    main()
