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
    return np.round(vals)


def main():
    # Set problem parameters
    n = 6
    # Generate a random 3-regular graph on n nodes
    graph = networkx.random_regular_graph(3, n)

    # Compute best possible cut value via brute force search
    cuts = brute_force(graph, n)
    print(cuts)

    # Make qubits
    circuit = cirq.Circuit()
    qubits = cirq.LineQubit.range(n)

    # Prepare uniform superposition
    circuit.append(cirq.H.on_each(*qubits))

    theta = 0.1 * np.pi / 2.0
    for i, j in graph.edges:
        circuit.append(cirq.CNOT(qubits[i], qubits[j]))
        circuit.append(cirq.Rz(rads=theta).on(qubits[j]))
        circuit.append(cirq.CNOT(qubits[i], qubits[j]))

    phi = cirq.Simulator().simulate(circuit, qubit_order=qubits, initial_state=0).state_vector()
    alpha = np.round(np.arctan2(phi.real, phi.imag) / theta * 2.0) / 2.0

    print(alpha)

    circuit.append(cirq.qft(*qubits))
    phi = cirq.Simulator().simulate(circuit, qubit_order=qubits, initial_state=0).state_vector()

    print(phi)


if __name__ == '__main__':
    main()
