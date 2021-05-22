# The effect of data encoding on the expressive power of variational quantum machine learning models
# Maria Schuld, Ryan Sweke, Johannes Jakob Meyer
# https://arxiv.org/abs/2008.08605

# Quantum embeddings for machine learning
# Seth Lloyd, Maria Schuld, Aroosa Ijaz, Josh Izaac, Nathan Killoran
# https://arxiv.org/abs/2001.03622

import cirq

def build_1d_circuit(x: float, theta: float, depth: int):
    circuit = cirq.Circuit()
    q = cirq.NamedQubit('q')
    for _ in range(depth):
        circuit.append(cirq.Rx(rads=x).on(q))
        circuit.append(cirq.Ry(rads=theta).on(q))
    return circuit

circuit = build_1d_circuit(0.1, 0.2, 3)

print(circuit)
