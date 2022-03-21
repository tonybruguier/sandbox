import cirq
import cirq.contrib.bayesian_network as ccb

def diffuser(qubits):
    for q in qubits:
        yield cirq.H(q)
        yield cirq.X(q)

    yield cirq.Z(qubits[-1]).controlled_by(*qubits[0:-1])

    for q in qubits:
        yield cirq.X(q)
        yield cirq.H(q)

def oracle(qubits):
    # # This Oracle should favor |11111>
    # yield cirq.Z(qubits[-1]).controlled_by(*qubits[0:-1])

    # This Oracle should favor |???1?>
    yield cirq.Z(qubits[3])


qubits = cirq.LineQubit.range(5)
circuit = cirq.Circuit()

# for q in qubits:
#     circuit += cirq.H(q)

circuit += ccb.BayesianNetworkGate(
    [('q0', 0.3), ('q1', 0.6), ('q2', 0.3), ('q3', None), ('q4', None)],
    [(('q3'), ('q0', 'q1',), [0.01, 0.02, 0.03, 0.04]),
     (('q4'), ('q2', 'q3',), [0.40, 0.10, 0.70, 0.90])]).on(*qubits)

# Contrary to classical algorithms, adding extra iterations will worsen the results.
for _ in range(1):
    circuit += oracle(qubits)
    circuit += diffuser(qubits)

phi = cirq.Simulator().simulate(circuit, qubit_order=qubits, initial_state=0).state_vector()
probs = abs(phi) ** 2

for shift in range(5):
    total_prob = 0.0
    for i, prob in enumerate(probs):
        if (i >> ( 5 - 1 - shift)) & 0x01:
            total_prob += prob
    print(f'p(q{shift} = 1) = {total_prob}')


for shift in range(5):
    prob_num = 0.0
    prob_den = 0.0
    for i, prob in enumerate(probs):
        if (i >> ( 5 - 1 - 3)) & 0x01:
            prob_den += prob
            if (i >> ( 5 - 1 - shift)) & 0x01:
                prob_num += prob
    print(f'p(q{shift} = 1 | q3 = 1) = {prob_num / prob_den}')

#print(cirq.dirac_notation(phi))
