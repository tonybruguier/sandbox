# https://www.tensorflow.org/quantum/tutorials/hello_many_worlds
import cirq
import numpy as np
import tensorflow as tf
import tensorflow_quantum as tfq
import sympy

num_examples = 7

# Changed once per example:
beta = sympy.symbols('beta')
# Same for all the examples:
theta = sympy.symbols('theta')

qubit = cirq.GridQubit(0, 0)
model_circuit = cirq.Circuit(cirq.rz(beta)(qubit), cirq.ry(theta)(qubit))

print(model_circuit)

circuit_input = tf.keras.Input(shape=(), dtype=tf.string, name='circuit_input')
beta_input = tf.keras.Input(shape=(num_examples,),  dtype=tf.dtypes.float32, name='beta_input')
theta_input = tf.keras.Input(shape=(1,),  dtype=tf.dtypes.float32, name='theta_input')

operators = [cirq.Z(qubit)]
expectation_layer = tfq.layers.ControlledPQC(model_circuit, operators=operators)
expectation = expectation_layer([circuit_input, beta_input, theta_input])

model = tf.keras.Model(
    inputs=[circuit_input, beta_input, theta_input],
    outputs=[expectation])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.05),
    loss=tf.keras.losses.MeanSquaredError())

commands = np.array([[0] * num_examples], dtype=np.float32)
targets = np.array([[0] * num_examples], dtype=np.float32)

history = model.fit(x=[tfq.convert_to_tensor([cirq.Circuit()]), commands, tfq.convert_to_tensor(operators)],
                    y=targets,
                    epochs=1,
                    verbose=0)
