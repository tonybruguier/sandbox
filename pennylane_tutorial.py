# https://pennylane.ai/install.html?version=source
# pip install -e .


import pennylane as qml
from pennylane import numpy as np

# dev1 = qml.device("cirq.simulator", wires=1)

# @qml.qnode(dev1)
# def circuit1(params):
#     qml.RX(params[0], wires=0)
#     qml.RY(params[1], wires=0)
#     return qml.expval(qml.PauliZ(0))

# @qml.qnode(dev1)
# def circuit2(phi1, phi2):
#     qml.RX(phi1, wires=0)
#     qml.RY(phi2, wires=0)
#     return qml.expval(qml.PauliZ(0))

# print(circuit1([0.54, 0.12]))
# print(circuit2(0.54, 0.12))

# dcircuit1 = qml.grad(circuit1, argnum=0)
# print(dcircuit1([0.54, 0.12]))

# dcircuit2 = qml.grad(circuit2, argnum=[0, 1])
# https://pennylane.ai/qml/demos/tutorial_gaussian_transformation.html

# print(dcircuit2(0.54, 0.12))

# def cost(x):
#     return circuit1(x)

# init_params = np.array([0.011, 0.012])
# print(cost(init_params))


# # initialise the optimizer
# opt = qml.GradientDescentOptimizer(stepsize=0.4)

# # set the number of steps
# steps = 100
# # set the initial parameter values
# params = init_params

# for i in range(steps):
#     # update the circuit parameters
#     params = opt.step(cost, params)

#     if (i + 1) % 5 == 0:
#         print("Cost after step {:5d}: {: .7f}".format(i + 1, cost(params)))

# print("Optimized rotation angles: {}".format(params))





# dev = qml.device("default.gaussian", wires=1)

# @qml.qnode(dev)
# def mean_photon_gaussian(mag_alpha, phase_alpha, phi):
#     qml.Displacement(mag_alpha, phase_alpha, wires=0)
#     qml.Rotation(phi, wires=0)
#     return qml.expval(qml.NumberOperator(0))

# def cost(params):
#     return (mean_photon_gaussian(params[0], params[1], params[2]) - 1.0) ** 2

# init_params = [0.015, 0.02, 0.005]
# print(cost(init_params))

# # initialise the optimizer
# opt = qml.GradientDescentOptimizer(stepsize=0.1)

# # set the number of steps
# steps = 20
# # set the initial parameter values
# params = init_params

# for i in range(steps):
#     # update the circuit parameters
#     params = opt.step(cost, params)

#     print("Cost after step {:5d}: {:8f}".format(i + 1, cost(params)))

# print("Optimized mag_alpha:{:8f}".format(params[0]))
# print("Optimized phase_alpha:{:8f}".format(params[1]))
# print("Optimized phi:{:8f}".format(params[2]))


# create the devices
dev_qubit = qml.device("default.qubit", wires=1)
dev_fock = qml.device("strawberryfields.fock", wires=2, cutoff_dim=10)


@qml.qnode(dev_fock, diff_method="parameter-shift")
def photon_redirection(params):
    qml.FockState(1, wires=0)
    qml.Beamsplitter(params[0], params[1], wires=[0, 1])
    return qml.expval(qml.NumberOperator(1))
