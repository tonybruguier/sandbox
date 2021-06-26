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

import tensorflow as tf
from tensorflow_quantum.python import util


import quantum_embed

class QuantumEmbedTest(tf.test.TestCase):
    """Tests for the QuantumEmbed layer."""

    def test_quantum_embed_instantiate(self):
        """Basic creation test."""
        qubit = cirq.GridQubit(0, 0)
        quantum_embed.QuantumEmbed([qubit])

    def test_pqc_double_learn(self):
        """Test a simple learning scenario using analytic and sample expectation
        on many backends."""
        qubit = cirq.GridQubit(0, 0)

        quantum_datum = tf.keras.Input(shape=(), dtype=tf.dtypes.string)
        qe = quantum_embed.QuantumEmbed([qubit])
        outputs = qe(quantum_datum)

        model = tf.keras.Model(inputs=quantum_datum, outputs=outputs)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.03),
                      loss=tf.keras.losses.mean_squared_error)

        data_out = np.array([[1], [-1]], dtype=np.float32)

        data_circuits = util.convert_to_tensor(
            [cirq.Circuit(cirq.X(qubit)**0.5),
             cirq.Circuit()])

        history = model.fit(x=data_circuits, y=data_out, epochs=40)
        self.assertAllClose(history.history['loss'][-1], 0, atol=1e-1)

if __name__ == "__main__":
    tf.test.main()
