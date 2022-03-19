import cirq

def diffuser(qubits):
    for q in qubits:
        yield cirq.H(q)
        yield cirq.X(q)

    yield cirq.Z(qubits[0]).controlled_by(*qubits[1:])

    for q in qubits:
        yield cirq.X(q)
        yield cirq.H(q)

def oracle(qubits):
    yield cirq.Z(qubits[0]).controlled_by(*qubits[1:])

qubits = cirq.LineQubit.range(3)
circuit = cirq.Circuit()

for q in qubits:
    circuit += cirq.H(q)

for _ in range(1):
    circuit += oracle(qubits)
    circuit += diffuser(qubits)

print(circuit)

phi = cirq.Simulator().simulate(circuit, qubit_order=qubits, initial_state=0).state_vector()

print(phi.round(decimals=4))
