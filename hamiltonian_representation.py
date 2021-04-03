import math

import numpy as np
from sympy.parsing.sympy_parser import parse_expr
from sympy.logic.boolalg import And, Not, Or, Xor
from sympy.core.symbol import Symbol
from typing import Dict, Tuple

import cirq

# References:
# [1] On the representation of Boolean and real functions as Hamil tonians for quantum computing
#     by StuartHadfield
# [2] https://www.youtube.com/watch?v=AOKM9BkweVU is a useful intro
# [3] https://github.com/rsln-s/IEEE_QW_2020/blob/master/Slides.pdf

class HamiltonianList():
    """A container class of Boolean function as equation (2) or [1]"""
    def __init__(self, hamiltonians: Dict[Tuple[int, ...], float]):
        self.hamiltonians = {h: w for h, w in hamiltonians.items() if math.fabs(w) > 1e-12}

    def __str__(self):
        return "; ".join("%.2f.%s" % (w, ".".join("Z_%d" % (d) for d in h) if h else 'I') for h, w in self.hamiltonians.items())

    def __add__(self, other):
        return self._signed_add(other, 1.0)

    def __sub__(self, other):
        return self._signed_add(other, -1.0)

    def _signed_add(self, other, sign: float):
        hamiltonians = self.hamiltonians.copy()
        for h, w in other.hamiltonians.items():
            if h not in hamiltonians:
                hamiltonians[h] = sign * w
            else:
                hamiltonians[h] += sign * w
        return HamiltonianList(hamiltonians)

    def __rmul__(self, other: float):
        return HamiltonianList({k : other * w for k, w in self.hamiltonians.items()})

    def __mul__(self, other):
        hamiltonians = {}
        for h1, w1 in self.hamiltonians.items():
            for h2, w2 in other.hamiltonians.items():
                h = tuple(set(h1).symmetric_difference(h2))
                w = w1 * w2
                if h not in hamiltonians:
                    hamiltonians[h] = w
                else:
                    hamiltonians[h] += w
        return HamiltonianList(hamiltonians)

    @staticmethod
    def O():
        return HamiltonianList({})

    @staticmethod
    def I():
        return HamiltonianList({(): 1.0})

    @staticmethod
    def Z(i: int):
        return HamiltonianList({(i, ): 1.0})


def build_hamiltonian_from_boolean(boolean_expr, name_to_id) -> HamiltonianList:
    """Builds the Hamiltonian representation of Boolean expression as per [1]:

    Args:
        boolean_expr: A Sympy expression containing symbols and Boolean operations
        name_to_id: A dictionary from symbol name to an integer, typically built by calling
            get_name_to_id().

    Return:
        The HamiltonianList that represents the Boolean expression.
    """

    if isinstance(boolean_expr, (And, Not, Or, Xor)):
        sub_hamiltonians = [build_hamiltonian_from_boolean(sub_boolean_expr, name_to_id) for sub_boolean_expr in boolean_expr.args]
        # We apply the equalities of theorem 1 of [1].
        if isinstance(boolean_expr, And):
            hamiltonian = HamiltonianList.I()
            for sub_hamiltonian in sub_hamiltonians:
                hamiltonian = hamiltonian * sub_hamiltonian
        elif isinstance(boolean_expr, Not):
            assert len(sub_hamiltonians) == 1
            hamiltonian = HamiltonianList.I() - sub_hamiltonians[0]
        elif isinstance(boolean_expr, Or):
            hamiltonian = HamiltonianList.O()
            for sub_hamiltonian in sub_hamiltonians:
                hamiltonian = hamiltonian + sub_hamiltonian - hamiltonian * sub_hamiltonian
        elif isinstance(boolean_expr, Xor):
            hamiltonian = HamiltonianList.O()
            for sub_hamiltonian in sub_hamiltonians:
                hamiltonian = hamiltonian + sub_hamiltonian - 2.0 * hamiltonian * sub_hamiltonian
        return hamiltonian
    elif isinstance(boolean_expr, Symbol):
        # Table 1 of [1], entry for 'x' is '1/2.I - 1/2.Z'
        i = name_to_id[boolean_expr.name]
        return 0.5 * HamiltonianList.I() - 0.5 * HamiltonianList.Z(i)
    else:
        raise ValueError('Unsupported type: %s' % (type(boolean_expr)))

def get_name_to_id(boolean_expr):
    return {symbol.name: i for i, symbol in enumerate(boolean_expr.free_symbols)}

def build_circuit_from_hamiltonian(hamiltonian, name_to_id, theta):
    qubits = [cirq.NamedQubit(name) for name in name_to_id.keys()]
    circuit = cirq.Circuit()

    circuit.append(cirq.H.on_each(*qubits))

    for h, w in hamiltonian.hamiltonians.items():
        for i in range(1, len(h)):
            circuit.append(cirq.CNOT(qubits[h[i]], qubits[h[0]]))

        if len(h) >= 1:
            circuit.append(cirq.Rz(rads=(theta * w)).on(qubits[h[0]]))

        for i in range(1, len(h)):
            circuit.append(cirq.CNOT(qubits[h[i]], qubits[h[0]]))

    return circuit, qubits

boolean = parse_expr('q0 & ~q1 & q2')
name_to_id = get_name_to_id(boolean)
hamiltonian = build_hamiltonian_from_boolean(boolean, name_to_id)

print(hamiltonian)

theta = 0.1 * math.pi
circuit, qubits = build_circuit_from_hamiltonian(hamiltonian, name_to_id, theta)

print(circuit)

phi = cirq.Simulator().simulate(circuit, qubit_order=qubits, initial_state=0).state_vector()
alpha = np.arctan2(phi.real, phi.imag) - math.pi / 2.0
print(alpha)