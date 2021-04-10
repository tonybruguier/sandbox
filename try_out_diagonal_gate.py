import cirq
import numpy as np

circuit = cirq.Circuit()
qubits = cirq.LineQubit.range(4)

op = cirq.DiagonalGate([0.25 * np.pi, 0.25 * np.pi, 0.5 * np.pi, 0.5 * np.pi])

print(np.diag(cirq.protocols.unitary(op)))
circuit.append(cirq.decompose(op.on(qubits[0], qubits[1])))

print(circuit)
