# The effect of data encoding on the expressive power of variational quantum machine learning models
# Maria Schuld, Ryan Sweke, Johannes Jakob Meyer
# https://arxiv.org/abs/2008.08605

# Quantum embeddings for machine learning
# Seth Lloyd, Maria Schuld, Aroosa Ijaz, Josh Izaac, Nathan Killoran
# https://arxiv.org/abs/2001.03622

# Supervised quantum machine learning models are kernel methods
# Maria Schuld
# https://arxiv.org/abs/2101.11020

# Also useful to get started:
# https://www.youtube.com/watch?v=mNR-7OmilIo


import cirq
import numpy as np

def build_parametrized_unitary(qubits, n_layers, theta):
    # Circuit-centric quantum classifiers
    # Maria Schuld, Alex Bocharov, Krysta Svore, Nathan Wiebe
    # https://arxiv.org/abs/1804.00633

    # PennyLane StronglyEntanglingLayers:
    # https://sourcegraph.com/github.com/PennyLaneAI/pennylane/-/blob/pennylane/templates/layers/strongly_entangling.py
    if len(qubits) > 1:
        ranges = [(l % (len(qubits) - 1)) + 1 for l in range(n_layers)]

    assert theta.shape[0] == n_layers
    assert theta.shape[1] == len(qubits)

    for l in range(n_layers):
        for i in range(len(qubits)):
            yield cirq.Rz(rads=theta[l][i][0]).on(qubits[i])
            yield cirq.Ry(rads=theta[l][i][1]).on(qubits[i])
            yield cirq.Rz(rads=theta[l][i][2]).on(qubits[i])

        if len(qubits) > 1:
            for i in range(len(qubits)):
                j = (i + ranges[l]) % len(qubits)
                yield cirq.CNOT(qubits[i], qubits[j])


def build_param_rotator(qubits, x):
    for qubit in qubits:
        yield cirq.Rx(rads=x).on(qubit)

n_layers = 2
qubits = cirq.LineQubit.range(3)
theta = np.random.rand(n_layers, len(qubits), 3)
x = 0.123 * np.pi


circuit = cirq.Circuit()
for gate in build_parametrized_unitary(qubits, n_layers, theta):
    circuit.append(gate)
for gate in build_param_rotator(qubits, x):
    circuit.append(gate)

print(circuit)
