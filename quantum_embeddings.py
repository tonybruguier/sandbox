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

def build_arbitrary_unitary(qubits, n_layers):
    # Circuit-centric quantum classifiers
    # Maria Schuld, Alex Bocharov, Krysta Svore, Nathan Wiebe
    # https://arxiv.org/abs/1804.00633

    # PennyLane StronglyEntanglingLayers:
    # https://sourcegraph.com/github.com/PennyLaneAI/pennylane/-/blob/pennylane/templates/layers/strongly_entangling.py
    if len(qubits) > 1:
        ranges = [(l % (len(qubits) - 1)) + 1 for l in range(n_layers)]
    else:
        ranges [0] * n_layers

    for l in range(n_layers):
        for i in range(len(qubits)):
            yield cirq.Rx(rads=0.1).on(qubits[i])
            yield cirq.Ry(rads=0.1).on(qubits[i])
            yield cirq.Rz(rads=0.1).on(qubits[i])

        if len(qubits) > 0:
            for i in range(len(qubits)):
                j = (i + ranges[l]) % len(qubits)
                yield cirq.CNOT(qubits[i], qubits[j])


qubits = cirq.LineQubit.range(8)

circuit = cirq.Circuit()
for gate in build_arbitrary_unitary(qubits, 3):
    circuit.append(gate)

# def build_1d_circuit(x: float, theta: float, depth: int):
#     circuit = cirq.Circuit()
#     q = cirq.NamedQubit('q')
#     for _ in range(depth):
#         circuit.append(cirq.Rx(rads=x).on(q))
#         circuit.append(cirq.Ry(rads=theta).on(q))
#     return circuit, [q]

# circuit, qubits = build_1d_circuit(0.1, 0.2, 3)

print(circuit)
