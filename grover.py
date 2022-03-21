import cirq
import cirq.contrib.bayesian_network as ccb
from matplotlib import pyplot as plt
import numpy as np

def diffuser(qubits):
    #yield cirq.X(qubits[0])
    for q in qubits:
        yield cirq.H(q)
        yield cirq.X(q)

    yield cirq.Z(qubits[-1]).controlled_by(*qubits[0:-1])

    for q in qubits:
        yield cirq.X(q)
        yield cirq.H(q)
    yield -1.0 * cirq.I(qubits[0])

def oracle(qubits):
    # # This Oracle should favor |11111>
    # yield cirq.Z(qubits[-1]).controlled_by(*qubits[0:-1])

    # This Oracle should favor |???1?>
    yield cirq.Z(qubits[3])

def print_state_vector(circuit):
    phi = cirq.Simulator().simulate(circuit, qubit_order=qubits, initial_state=0).state_vector()

    print('-----')
    print('01234')
    print('-----')
    for i in range(2**5):
        print("{i:05b}\tp={p:.3f}\t({real:.6f},\t{imag:.6f})".format(
          i=i, p=abs(phi[i])**2, real=phi[i].real, imag=phi[i].imag))
    print("\t\t({real:.6f},\t{imag:.6f})".format(
        real=np.mean(phi).real, imag=np.mean(phi).imag))

qubits = cirq.LineQubit.range(5)
circuit = cirq.Circuit()

# for q in qubits:
#     circuit += cirq.H(q)

circuit += ccb.BayesianNetworkGate(
    [('q0', 0.3), ('q1', 0.6), ('q2', 0.3), ('q3', None), ('q4', None)],
    [(('q3'), ('q0', 'q1',), [0.01, 0.02, 0.03, 0.04]),
     (('q4'), ('q2', 'q3',), [0.40, 0.10, 0.70, 0.90])]).on(*qubits)

print_state_vector(circuit)

# Contrary to classical algorithms, adding extra iterations will worsen the results.
x = cirq.Simulator().simulate(circuit, qubit_order=qubits, initial_state=0).state_vector()


for _ in range(1):
    circuit += oracle(qubits)
    print_state_vector(circuit)

    circuit += diffuser(qubits)
    print_state_vector(circuit)


y = cirq.Simulator().simulate(circuit, qubit_order=qubits, initial_state=0).state_vector()

idx_q3_0 = []
idx_q3_1 = []
for i in range(2 ** 5):
    if (i >> ( 5 - 1 - 3)) & 0x01:
        idx_q3_1.append(i)
    else:
        idx_q3_0.append(i)

plt.plot(x[idx_q3_0], y[idx_q3_0], '.r')
plt.plot(x[idx_q3_1], y[idx_q3_1], '.b')
plt.grid()
plt.legend(['q3=0', 'q3=1'])
plt.savefig('grover.png')


#print(cirq.unitary(cirq.Circuit(diffuser(qubits))))

# for shift in range(5):
#     total_prob = 0.0
#     for i, prob in enumerate(probs):
#         if (i >> ( 5 - 1 - shift)) & 0x01:
#             total_prob += prob
#     print(f'p(q{shift} = 1) = {total_prob}')

probs = [abs(x) ** 2 for x in cirq.Simulator().simulate(circuit, qubit_order=qubits, initial_state=0).state_vector()]
for shift in range(5):
    prob_num = 0.0
    prob_den = 0.0
    for i, prob in enumerate(probs):
        prob_den += prob
        if (i >> ( 5 - 1 - shift)) & 0x01:
            prob_num += prob
    print(f'p(q{shift} = 1) = {prob_num / prob_den}')

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
