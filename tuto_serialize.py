#import tensorflow as tf
import cirq
#import matplotlib.pyplot as plt
#import numpy as np
from tensorflow_quantum.core.ops.math_ops import simulate_mps
from tensorflow_quantum.python import util

qubit = cirq.GridQubit(1, 1)

circuit0 = cirq.Circuit()
circuit0 += cirq.PhasedXZGate(axis_phase_exponent=-0.5, x_exponent=0.5, z_exponent=-0.5)(qubit)

U0 = cirq.unitary(circuit0)


# $Z^z Z^a X^x Z^{-a}$.
for e1 in [-1.0, -0.5, 0.0, 0.5, 1.0]:
    for e2 in [-1.0, -0.5, 0.0, 0.5, 1.0]:
        for e3 in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            circuit1 = cirq.Circuit()
            circuit1 += cirq.ZPowGate(exponent=e1)(qubit)
            circuit1 += cirq.XPowGate(exponent=e2)(qubit)
            circuit1 += cirq.ZPowGate(exponent=e3)(qubit)

            U1 = cirq.unitary(circuit1)

            if (U0 == U1).all():
                print(f'TONYBOOM e1={e1} e2={e2} e3={e3}')

# TONYBOOM e1=0.5 e2=0.5 e3=-1.0
# TONYBOOM e1=0.5 e2=0.5 e3=1.0

programs=util.convert_to_tensor([circuit1])
