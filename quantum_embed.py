# Copyright 2021 The TensorFlow Quantum Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import cirq
import numpy as np
import sympy
import tensorflow as tf

from tensorflow_quantum.python import util
from tensorflow_quantum.python.layers.circuit_construction import elementary
from tensorflow_quantum.python.layers.circuit_executors import expectation


class QuantumEmbed(tf.keras.layers.Layer):
    """Quantum Embdedding Layer.

    This layers emed classical data according the papers:

    The effect of data encoding on the expressive power of variational quantum machine learning models
    Maria Schuld, Ryan Sweke, Johannes Jakob Meyer
    https://arxiv.org/abs/2008.08605

    Quantum embeddings for machine learning
    Seth Lloyd, Maria Schuld, Aroosa Ijaz, Josh Izaac, Nathan Killoran
    https://arxiv.org/abs/2001.03622

    Supervised quantum machine learning models are kernel methods
    Maria Schuld
    https://arxiv.org/abs/2101.11020

    Also useful to get started:
    https://www.youtube.com/watch?v=mNR-7OmilIo
    """

    def __init__(self, qubits, num_repetitions_input, depth_input,
                 num_unitary_layers, num_repetitions, **kwargs) -> None:
        """Instantiate this layer."""
        super().__init__(**kwargs)

        assert len(qubits) == num_repetitions_input
        assert len(qubits[0]) == depth_input

        self._theta = np.random.uniform(
            0, 2 * np.pi, (num_repetitions_input, depth_input,
                           num_unitary_layers, num_repetitions, 3))

        self._model_circuits = []
        for l in range(num_repetitions):
            circuit = cirq.Circuit(
                _build_parametrized_unitary(qubits, depth_input,
                                            num_repetitions_input,
                                            num_unitary_layers,
                                            self._theta[:, :, :, l, :]))
            self._model_circuits.append(util.convert_to_tensor([circuit]))

        self._operators = util.convert_to_tensor([[cirq.Z(qubits[0][0])]])
        self._executor = expectation.Expectation(backend='noiseless',
                                                 differentiator=None)
        self._append_layer = elementary.AddCircuit()

    @property
    def symbols(self):
        pass

    def call(self, inputs):
        """Keras call function."""

        circuit_batch_dim = tf.gather(tf.shape(inputs), 0)

        model_appended = tf.tile(util.convert_to_tensor([cirq.Circuit()]),
                                 [circuit_batch_dim])
        for model_circuit in self._model_circuits:
            tiled_up_model = tf.tile(model_circuit, [circuit_batch_dim])
            model_appended = self._append_layer(model_appended, append=inputs)
            model_appended = self._append_layer(model_appended,
                                                append=tiled_up_model)

        tiled_up_operators = tf.tile(self._operators, [circuit_batch_dim, 1])
        return self._executor(model_appended, operators=tiled_up_operators)


def _build_param_rotator(qubits, depth_input, num_repetitions_input, x):
    assert x.shape == (depth_input,)
    assert len(qubits) == num_repetitions_input
    assert len(qubits[0]) == depth_input

    for i in range(num_repetitions_input):
        for j in range(depth_input):
            yield cirq.Rx(rads=x[j]).on(qubits[i][j])


def _build_parametrized_unitary(qubits, depth_input, num_repetitions_input,
                                num_unitary_layers, theta):
    # Circuit-centric quantum classifiers
    # Maria Schuld, Alex Bocharov, Krysta Svore, Nathan Wiebe
    # https://arxiv.org/abs/1804.00633

    # PennyLane StronglyEntanglingLayers:
    # https://sourcegraph.com/github.com/PennyLaneAI/pennylane/-/blob/pennylane/templates/layers/strongly_entangling.py
    assert theta.shape == (
        num_repetitions_input,
        depth_input,
        num_unitary_layers,
        3,
    )

    assert len(qubits) == num_repetitions_input
    assert len(qubits[0]) == depth_input

    num_qubits = depth_input * num_repetitions_input

    if num_qubits > 1:
        ranges = [(k % (num_qubits - 1)) + 1 for k in range(num_unitary_layers)]

    for k in range(num_unitary_layers):
        for i in range(num_repetitions_input):
            for j in range(depth_input):
                yield cirq.Rz(rads=theta[i][j][k][0]).on(qubits[i][j])
                yield cirq.Ry(rads=theta[i][j][k][1]).on(qubits[i][j])
                yield cirq.Rz(rads=theta[i][j][k][2]).on(qubits[i][j])

        if num_qubits > 1:
            for ij1 in range(num_qubits):
                ij2 = (ij1 + ranges[k]) % num_qubits
                assert ij1 != ij2

                i1 = ij1 % num_repetitions_input
                j1 = ij1 // num_repetitions_input

                i2 = ij2 % num_repetitions_input
                j2 = ij2 // num_repetitions_input

                yield cirq.CNOT(qubits[i1][j1], qubits[i2][j2])
