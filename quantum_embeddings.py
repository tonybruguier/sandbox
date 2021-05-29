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

import math
import numpy as np
import scipy.optimize

import cirq

def build_parametrized_unitary(qubits, N, theta):
    # Circuit-centric quantum classifiers
    # Maria Schuld, Alex Bocharov, Krysta Svore, Nathan Wiebe
    # https://arxiv.org/abs/1804.00633

    # PennyLane StronglyEntanglingLayers:
    # https://sourcegraph.com/github.com/PennyLaneAI/pennylane/-/blob/pennylane/templates/layers/strongly_entangling.py
    if len(qubits) > 1:
        ranges = [(l % (len(qubits) - 1)) + 1 for l in range(N)]

    assert theta.shape[0] == N
    assert theta.shape[1] == len(qubits)
    assert theta.shape[2] == 3

    for l in range(N):
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

def build_fidelity_swap_circuit(qubit, ancilla_fidelity, ancilla_truth, key):
    # https://staff.fnwi.uva.nl/m.walter/physics491/lecture10.pdf
    yield cirq.H(ancilla_fidelity)
    yield cirq.CSWAP(ancilla_fidelity, qubit, ancilla_truth)
    yield cirq.H(ancilla_fidelity)
    yield cirq.measure(ancilla_fidelity, key=f"swap_test_{key}")

def build_embed_circuit(qubits, ancillas_fidelity, ancillas_truth, x, y, theta):
    Np, Dp, Nu, Du, _ = theta.shape
    assert Dp == len(qubits)
    assert Du == len(qubits[0])
    assert Dp == x.shape[0]
    assert Dp == y.shape[0]

    for du in range(Du):
        for dp in range(Dp):
            yield cirq.Ry(rads=(math.pi * y[dp])).on(ancillas_truth[dp][du])

    for np in range(Np):
        for dp in range(Dp):
            # Add a unitary:
            yield build_parametrized_unitary(qubits[dp], Nu, theta[np, dp])
            # Add the encoder:
            yield build_param_rotator(qubits[dp], x[dp])

    for du in range(Du):
        for dp in range(Dp):
            yield build_fidelity_swap_circuit(qubits[dp][du], ancillas_fidelity[dp][du], ancillas_truth[dp][du], f"p{dp}_u{du}")

# Some constants and qubits.
Nu = 1  # Number of layers inside the unitary
Du = 2  # Depth of the unitary:
Np = 3  # Number of times the input is fed:
Dp = 2  # Depth of the input

qubits = [[cirq.NamedQubit(f"q_p{dp}_u{du}") for du in range(Du)] for dp in range(Dp)]
ancillas_fidelity = [[cirq.NamedQubit(f"af_p{dp}_u{du}") for du in range(Du)] for dp in range(Dp)]
ancillas_truth = [[cirq.NamedQubit(f"at_p{dp}_u{du}") for du in range(Du)] for dp in range(Dp)]

def f(theta):
    theta = np.reshape(theta, (Np, Dp, Nu, Du, 3))

    x = np.random.normal(size=(2))
    y = np.asarray([1.0 if x[0] ** 2 + x[1] ** 2 < 0.5 else 0.0, 0.0])

    circuit = cirq.Circuit()
    for gate in build_embed_circuit(qubits, ancillas_fidelity, ancillas_truth, x, y, theta):
        circuit.append(gate)

    simulator = cirq.Simulator()
    run = simulator.run(circuit, repetitions=1000)

    estimated_fidelity = 1.0 - 2.0 * np.average([np.average(x) for x in run.measurements.values()])

    print('%.3f' % (estimated_fidelity))

    return -estimated_fidelity

theta0 = np.random.normal(size=(Np * Dp *  Nu * Du * 3))
scipy.optimize.minimize(f, theta0, method='Nelder-Mead', options={'maxiter': 100})
