import math

from sympy.parsing.sympy_parser import parse_expr
from sympy.logic.boolalg import And, Not, Or, Xor
from sympy.core.symbol import Symbol
from typing import Dict, Tuple

class HamiltonianList():
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


def build_hamiltonian_from_boolean(boolean_expr, name_to_id  = None) -> HamiltonianList:
    """Builds the Hamiltonian representation of Boolean expression as per:
    On the representation of Boolean and real functions as Hamil tonians for quantum computing
    StuartHadfield
    """
    if not name_to_id:
        name_to_id = {symbol.name: i for i, symbol in enumerate(boolean_expr.free_symbols)}

    if isinstance(boolean_expr, (And, Not, Or, Xor)):
        sub_hamiltonians = [build_hamiltonian_from_boolean(sub_boolean_expr, name_to_id) for sub_boolean_expr in boolean_expr.args]
        # We apply the equalities of theorem 1
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
        # Table 1, entry for 'x' is '1/2.I - 1/2.Z'
        i = name_to_id[boolean_expr.name]
        return 0.5 * HamiltonianList.I() - 0.5 * HamiltonianList.Z(i)
    else:
        raise ValueError('Unsupported type: %s' % (type(boolean_expr)))


boolean = parse_expr('~q0')
hamiltonian = build_hamiltonian_from_boolean(boolean)

print(hamiltonian)
