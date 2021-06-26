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

    def __init__(self, qubits, **kwargs) -> None:
        """Instantiate this layer."""
        super().__init__(**kwargs)

        model_circuits = [cirq.Circuit()]
        self._model_circuits = [
            util.convert_to_tensor([model_circuit])
            for model_circuit in model_circuits
        ]

        self._operators = util.convert_to_tensor([[cirq.Z(qubits[0])]])
        self._executor = expectation.Expectation(backend='noiseless', differentiator=None)
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
            model_appended = self._append_layer(model_appended, append=tiled_up_model)

        tiled_up_operators = tf.tile(self._operators, [circuit_batch_dim, 1])
        return self._executor(model_appended, operators=tiled_up_operators)

