import cirq
import numpy as np
from sympy.parsing.sympy_parser import parse_expr
import examples.hamiltonian_representation as hr
import math

qubits = [cirq.NamedQubit(name) for name in ['q0', 'q1']]
theta = 0.1 * math.pi

# Diagonal gate:
circuit1 = cirq.Circuit()
angles = [theta/4.0, -theta/4.0, -theta/4.0, theta/4.0]
circuit1.append(cirq.decompose(cirq.DiagonalGate(angles).on(*qubits)))
print(circuit1)
print(np.diag(cirq.unitary(circuit1)))

# Hamiltonian:
circuit2 = cirq.Circuit()

boolean = parse_expr('q0 ^ q1')
name_to_id = hr.get_name_to_id([boolean])
hamiltonian = hr.build_hamiltonian_from_boolean(boolean, name_to_id)

circuit2 += hr.build_circuit_from_hamiltonians([hamiltonian], qubits, theta)
print(circuit2)
print(np.diag(cirq.unitary(circuit2)))
