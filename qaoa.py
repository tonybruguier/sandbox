import itertools
import math

import numpy as np
import networkx
import scipy.optimize
from sympy.parsing.sympy_parser import parse_expr

import cirq
import examples.hamiltonian_representation as hr

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

    # Build the boolean expressions
    booleans = [parse_expr(f"x{i} ^ x{j}") for i, j in graph.edges]

    name_to_id = hr.get_name_to_id(booleans)
    hamiltonians = [hr.build_hamiltonian_from_boolean(boolean, name_to_id) for boolean in booleans]

    qubits = [cirq.NamedQubit(name) for name in name_to_id.keys()]
    circuit = cirq.Circuit()
    circuit.append(cirq.H.on_each(*qubits))

    theta = 0.1 * math.pi
    circuit += hr.build_circuit_from_hamiltonians(hamiltonians, qubits, theta)

    print(circuit)

if __name__ == '__main__':
    main()
