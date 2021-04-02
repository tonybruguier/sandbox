from sympy.parsing.sympy_parser import parse_expr
from sympy.logic.boolalg import And, Or
from sympy.core.symbol import Symbol

import cirq

def parse_boolean(boolean_expr, circuit, qubits = None):
    if not qubits:
        qubits = {}

    if isinstance(boolean_expr, (And, Or)):
        operands = [parse_boolean(sub_boolean_expr, circuit, qubits) for sub_boolean_expr in boolean_expr.args]
        if isinstance(boolean_expr, And):
            pass
        elif isinstance(boolean_expr, Or):
            pass
    elif isinstance(boolean_expr, Symbol):
        if boolean_expr.name not in qubits.keys():
            new_qubit = cirq.NamedQubit(boolean_expr.name)
            circuit.append(cirq.H.on(new_qubit))
            qubits[boolean_expr.name] = new_qubit
        return qubits[boolean_expr.name]
    else:
        raise ValueError('Unsupported type: %s' % (type(boolean_expr)))

circuit = cirq.Circuit()
expr = parse_expr('(q0 & q1 & q3) | q2')

parse_boolean(expr, circuit)

print(circuit)
