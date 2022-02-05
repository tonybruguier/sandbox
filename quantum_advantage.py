import tensorflow as tf
import cirq
import matplotlib.pyplot as plt
import numpy as np
import sympy

def un_bell_pair_block(qubits):
    return [cirq.CNOT(qubits[0], qubits[1]), cirq.H(qubits[0])]

def inv_z_basis_gate(pauli):
    if pauli == "I" or pauli == "Z":
        return cirq.I
    if pauli == "X":
        return cirq.H
    if pauli == "Y":
        # S^dag H to get to computational, H S to go back.
        return cirq.PhasedXZGate(
            axis_phase_exponent=-0.5, x_exponent=0.5, z_exponent=-0.5
        )
    raise ValueError("Invalid Pauli.")

def build_circuit(
    qubit_pairs,
    pauli,
    n_shots,
    rand_state,
    classical_shadows):
    a_qubits = [pair[0] for pair in qubit_pairs]
    b_qubits = [pair[1] for pair in qubit_pairs]
    all_qubits = np.concatenate(qubit_pairs)

    flip_params = sympy.symbols(f"param_0:{len(qubit_pairs) * 2}")

    # Add X flips.
    ret_circuit = cirq.Circuit(cirq.X(q) ** p for q, p in zip(all_qubits, flip_params))

    # Add basis turns a and b.
    ret_circuit += [
        inv_z_basis_gate(p)(q) for q, p in zip(a_qubits, pauli)
    ]
    ret_circuit += [
        inv_z_basis_gate(p)(q) for q, p in zip(b_qubits, pauli)
    ]

    if classical_shadows:
        # Add measurements.
        for i, qubit in enumerate(a_qubits):
            ret_circuit += cirq.measure(qubit, key=f"q{i}")
    else:  # not classical_shadows
        # Add un-bell pair.
        ret_circuit += [un_bell_pair_block(pair) for pair in qubit_pairs]

        # Add measurements.
        for i, qubit in enumerate(all_qubits):
            ret_circuit += cirq.measure(qubit, key=f"q{i}")

    # Create randomized flippings. These flippings will contain values of 1,0.
    # which will turn the X gates on or off.
    params = create_randomized_sweeps(
        pauli, flip_params, n_shots, rand_state
    )
    return ret_circuit, params

def create_randomized_sweeps(
    hidden_p,
    symbols,
    n_params,
    rand_state):
    last_i = 0
    for i, pauli in enumerate(hidden_p):
        if pauli != "I":
            last_i = i

    sign_p = rand_state.choice([1, -1])
    all_sweeps = []
    for _ in range(n_params):
        current_sweep = dict()
        for twocopy in [0, 1]:
            parity = sign_p * rand_state.choice([1, -1], p=[0.95, 0.05])
            for i, pauli in enumerate(hidden_p):
                current_symbol = symbols[2 * i + twocopy]
                current_sweep[current_symbol] = rand_state.choice([0, 1])
                if pauli != "I":
                    if last_i == i:
                        v = 1 if parity == -1 else 0
                        current_sweep[current_symbol] = v
                    elif current_sweep[current_symbol] == 1:
                        parity *= -1

        all_sweeps.append(current_sweep)
    return all_sweeps

rand_source = np.random.RandomState(20160913)
n_paulis = 3
n = 3
n_shots = 11
n_repeats = 13
classical_shadows = True

system_pairs = [(cirq.GridQubit(0, i), cirq.GridQubit(1, i)) for i in range(n)]
simulator = cirq.Simulator()

all_results = []

if classical_shadows:
    qubit_order = [f"q{i}" for i in range(n)]
else:  # not classical_shadows
    qubit_order = [f"q{i}" for i in range(2 * n)]

for pauli_num in rand_source.choice(range(4 ** n), n_paulis, replace=False):
    pauli = ''
    for _ in range(n):
        base4 = pauli_num % 4
        if base4 == 0:
            pauli += 'I'
        elif base4 == 1:
            pauli += 'X'
        elif base4 == 2:
            pauli += 'Y'
        else:
            pauli += 'Z'
        pauli_num = (pauli_num - base4) // 4
    print(pauli)

    circuit, sweeps = build_circuit(
        system_pairs,
        pauli,
        n_shots,
        rand_source,
        classical_shadows=classical_shadows)

    results_for_pauli = []
    for _ in range(n_repeats):
        results_for_repeat = []
        results = simulator.run_sweep(
            program=circuit,
            params=sweeps,
            repetitions=1
        )

        batch_results = []
        for j, single_circuit_samples in enumerate(results):
            out0 = single_circuit_samples.data[qubit_order].to_numpy()
            batch_results.append(np.squeeze(out0))

        results_for_pauli.append(np.array(batch_results))
    all_results.append(results_for_pauli)


class InnerLayer(tf.keras.Model):
    def __init__(self, n_shots, num_qubits):
        super(InnerLayer, self).__init__(name='inner')
        self.n_shots = n_shots
        self.num_qubits = num_qubits
        self.gru1 = tf.keras.layers.GRU(4, go_backwards=False, return_sequences=True)
        self.gru2 = tf.keras.layers.GRU(4, go_backwards=True,  return_sequences=True)
        self.gru3 = tf.keras.layers.GRU(4, go_backwards=False, return_sequences=False)

    def call(self, x):
        x = tf.expand_dims(tf.reshape(x, (-1, self.num_qubits)), -1)
        x = self.gru1(x)
        x = self.gru2(x)
        x = self.gru3(x)
        x = tf.reshape(x, (-1, self.n_shots, 4))
        return x

class IntermediateLayer(tf.keras.Model):
    def __init__(self):
        super(IntermediateLayer, self).__init__(name='intermediate')

    def build(self, input_shape):
        self.kernel = self.add_weight("kernel", shape=[int(input_shape[2]), 8])

    def call(self, x):
        x = tf.math.reduce_mean(x, axis=1)
        x = tf.matmul(x, self.kernel)
        return x

model = tf.keras.Sequential()
model.add(InnerLayer(n_shots, len(qubit_order)))
model.add(IntermediateLayer())

input_1 = tf.keras.Input((n_shots, len(qubit_order),))
input_2 = tf.keras.Input((n_shots, len(qubit_order),))

encoded_1 = model(input_1)
encoded_2 = model(input_2)

class OuterLayer(tf.keras.Model):
    def __init__(self):
        super(OuterLayer, self).__init__(name='')

    def call(self, x):
        x = tf.norm(x[1] - x[0], ord=2, axis=1)
        x = tf.stack([x, tf.ones(tf.shape(x))], axis=1)
        x = tf.nn.softmax(x)
        return x

predictor = OuterLayer()
prediction = predictor([encoded_1, encoded_2])

conjoined_net = tf.keras.Model(inputs=[input_1, input_2], outputs=prediction)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)

conjoined_net.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

def _sample_different(max_val, ref):
    ret = ref
    while ret == ref:
        ret = rand_source.choice(max_val)
    return ret

x1 = []
x2 = []
y = []
for pauli_idx in range(n_paulis):
    # Same Pauli
    for i in range(n_repeats):
        j = _sample_different(n_repeats, i)

        x1.append(all_results[pauli_idx][i].astype(float))
        x2.append(all_results[pauli_idx][j].astype(float))
        y.append([1.0, 0.0])

    # Different Pauli
    for i in range(n_repeats):
        other_pauli_idx = _sample_different(n_paulis, pauli_idx)
        j = rand_source.choice(n_repeats)
        x1.append(all_results[pauli_idx][i].astype(float))
        x2.append(all_results[other_pauli_idx][j].astype(float))
        y.append([0.0, 1.0])

x1 = np.stack(x1)
x2 = np.stack(x2)
y = np.stack(y)

history = conjoined_net.fit(
    x=[x1, x2],
    y=y,
    epochs=500,
    batch_size=(2*n_paulis),
    verbose=0)


plt.plot(history.history['accuracy'], 'b')
plt.ylabel('accuracy')
plt.show()

plt.plot(history.history['loss'], 'r')
plt.ylabel('loss')
plt.show()
