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
import itertools
import math
import numpy as np
import tensorflow as tf
import tensorflow_quantum as tfq
import sympy

def build_parametrized_unitary(qubits, N, theta):
    # Circuit-centric quantum classifiers
    # Maria Schuld, Alex Bocharov, Krysta Svore, Nathan Wiebe
    # https://arxiv.org/abs/1804.00633

    # PennyLane StronglyEntanglingLayers:
    # https://sourcegraph.com/github.com/PennyLaneAI/pennylane/-/blob/pennylane/templates/layers/strongly_entangling.py
    if len(qubits) > 1:
        ranges = [(l % (len(qubits) - 1)) + 1 for l in range(N)]

    assert len(theta) == N
    assert len(theta[0]) == len(qubits)
    assert len(theta[0][0]) == 3

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
    # CSWAP cannot be serialize, so it is decomposed.
    yield cirq.decompose(cirq.CSWAP(ancilla_fidelity, qubit, ancilla_truth))
    yield cirq.H(ancilla_fidelity)

def build_embed_circuit(qubits, ancillas_fidelity, ancillas_truth, x, y, theta):
    Np = len(theta)
    Dp = len(theta[0])
    Nu = len(theta[0][0])
    Du = len(theta[0][0][0])
    assert Dp == len(qubits)
    assert Du == len(qubits[0])
    assert Dp == len(x)
    assert Dp == y.shape[0]

    for du in range(Du):
        for dp in range(Dp):
            yield cirq.Ry(rads=(math.pi * y[dp])).on(ancillas_truth[dp][du])

    for np in range(Np):
        for dp in range(Dp):
            # Add a unitary:
            yield build_parametrized_unitary(qubits[dp], Nu, theta[np][dp])
            # Add the encoder:
            yield build_param_rotator(qubits[dp], x[dp])

    for du in range(Du):
        for dp in range(Dp):
            yield build_fidelity_swap_circuit(qubits[dp][du], ancillas_fidelity[dp][du], ancillas_truth[dp][du], f"p{dp}_u{du}")

# Some constants and qubits.
Nu = 1  # Number of layers inside the unitary
Du = 1  # Depth of the unitary:
Np = 1  # Number of times the input is fed:
Dp = 2  # Depth of the input

qubits = [[cirq.GridQubit(du, dp) for du in range(Du)] for dp in range(Dp)]
ancillas_fidelity = [[cirq.GridQubit(du + Du, dp) for du in range(Du)] for dp in range(Dp)]
ancillas_truth = [[cirq.GridQubit(du + 2 * Du, dp) for du in range(Du)] for dp in range(Dp)]

theta = np.asarray([[[[[sympy.symbols(f"theta_{np}_{dp}_{nu}_{du}_{i}")
    for i in range(3)]
    for du in range(Du)]
    for nu in range(Nu)]
    for dp in range(Dp)]
    for np in range(Np)])

theta_vals = np.random.normal(size=(Np, Dp, Nu, Du, 3))

x = np.asarray([sympy.symbols(f"x_{i}") for i in range(Dp)])
x_vals = np.random.normal(size=(Dp))

y_vals = np.asarray([1.0 if sum(x_i**2 for x_i in x_vals) < 0.5 else 0.0, 0.0])

model_circuit = cirq.Circuit()
for gate in build_embed_circuit(qubits, ancillas_fidelity, ancillas_truth, x, y_vals, theta):
    model_circuit.append(gate)

# ---- Eval time ----
# operators = [cirq.Z(af) for af in itertools.chain(*ancillas_fidelity)]
# symbol_names = tuple(np.concatenate((x.reshape(-1), theta.reshape(-1),)))
# symbol_values = np.concatenate((x_vals.reshape(-1), theta_vals.reshape(-1),))
# symbol_values = np.expand_dims(symbol_values, 0)
# exp = tfq.layers.Expectation()(
#     model_circuit, symbol_names=symbol_names, symbol_values=symbol_values, operators=operators)

# ---- Train time ----
control_params = tuple(theta.reshape(-1))


circuits_input = tf.keras.Input(shape=(),
                                # The circuit-tensor has dtype `tf.string`
                                dtype=tf.string,
                                name='circuits_input')

commands_input = tf.keras.Input(shape=(Dp,),
                                dtype=tf.dtypes.float32,
                                name='commands_input')

operators = [cirq.Z(af) for af in itertools.chain(*ancillas_fidelity)]
expectation_layer = tfq.layers.ControlledPQC(model_circuit, operators=operators)
expectation = expectation_layer([circuits_input, commands_input])

full_circuit = tfq.layers.AddCircuit()(circuits_input, append=model_circuit)
model = tf.keras.Model(inputs=[circuits_input, commands_input], outputs=expectation)

tf.keras.utils.plot_model(model, show_shapes=True, dpi=70)

# simulator = cirq.Simulator()
# run = simulator.run(circuit, repetitions=1000)
# estimated_fidelity = 1.0 - 2.0 * np.average([np.average(x) for x in run.measurements.values()])

optimizer = tf.keras.optimizers.Adam(lr=0.1)
