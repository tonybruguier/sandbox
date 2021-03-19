import cirq

import numpy as np

# Based on:
# Stabilizer Codes and Quantum Error Correction
# Daniel Gottesman
# https://thesis.library.caltech.edu/2900/2/THESIS.pdf

class StabilitizerCode(object):
    def __init__(self, group_generators):
        n = len(group_generators[0])
        k = n - len(group_generators)

        # Build the matrix defined in section 3.4
        M = np.zeros((n - k, 2 * n), np.int8)
        for i, group_generator in enumerate(group_generators):
            for j, c in enumerate(group_generator):
                if c == 'X' or c == 'Y':
                    M[i, j] = 1
                elif c == 'Z' or c == 'Y':
                    M[i, n + j] = 1

        # Performing the Gaussian elimination as in section 4.1
        def _GaussianElimination(M, min_row, max_row, min_col, max_col):
            max_rank = min(max_row - min_row, max_col - min_col)

            rank = 0
            for r in range(max_rank):
                i = min_row + r
                j = min_col + r
                pivot_rows, pivot_cols = np.nonzero(M[i:max_row, j:max_col])

                if pivot_rows.size == 0:
                    break

                pi = pivot_rows[0]
                pj = pivot_cols[0]

                # Swap the rows and columns:
                M[[i, i + pi]] = M[[i + pi, i]]
                M[:, [(j + pj), j]] = M[:, [j, (j + pj)]]

                # Do the elimination.
                for k in range(i + 1, max_row):
                    if M[k, j] == 1:
                        M[k, :] = np.mod(M[i, :] + M[k, :], 2)

                rank += 1

            # Backward replacing to get identity
            for r in reversed(range(rank)):
                i = min_row + r
                j = min_col + r

                # Do the elimination.
                for k in range(min_row, i):
                    if M[k, j] == 1:
                        M[k, :] = np.mod(M[i, :] + M[k, :], 2)

            return rank

        r = _GaussianElimination(M, 0, n - k, 0, n)
        _ = _GaussianElimination(M, r, n - k, n + r, 2 * n)

        # Get matrix sub-components, as per equation 4.3:
        A1 = M[0:r, r : (n - k)]
        A2 = M[0:r, (n - k) : n]
        B = M[0:r, n : (n + r)]
        C1 = M[0:r, (n + r) : (2 * n - k)]
        C2 = M[0:r, (2 * n - k) : (2 * n)]
        D = M[r : (2 * r), n : (n + r)]
        E = M[r : (2 * r), (2 * n - k) : (2 * n)]

        X = np.concatenate(
            [
                np.zeros((k, r), dtype=np.int8),
                E.T,
                np.eye(k, dtype=np.int8),
                np.mod(E.T @ C1.T + C2.T, 2),
                np.zeros((k, n - k - r), np.int8),
                np.zeros((k, k), np.int8),
            ],
            axis=1,
        )

        Z = np.concatenate(
            [
                np.zeros((k, n), dtype=np.int8),
                A2.T,
                np.zeros((k, n - k - r), dtype=np.int8),
                np.eye(k, dtype=np.int8),
            ],
            axis=1,
        )

        self.n = n
        self.k = k
        self.r = r

        def _BuildByCode(mat):
            out = []
            n = mat.shape[1] // 2
            for i in range(mat.shape[0]):
                ps = ''
                for j in range(n):
                    if mat[i, j] == 0 and mat[i, j + n] == 0:
                        ps += 'I'
                    elif mat[i, j] == 1 and mat[i, j + n] == 0:
                        ps += 'X'
                    elif mat[i, j] == 0 and mat[i, j + n] == 1:
                        ps += 'Z'
                    else:
                        ps += 'Y'
                out.append(ps)
            return out

        self.M = _BuildByCode(M)
        self.X = _BuildByCode(X)
        self.Z = _BuildByCode(Z)
        self.syndromes_to_corrections = {}

        for qid in range(self.n):
            for op in ['X', 'Z']:
                syndrome = [1 if self.M[r][qid] == 'I' or self.M[r][qid] == op else -1 for r in range(self.r)]
                self.syndromes_to_corrections[tuple(syndrome)] = (op, qid)

    def encode(self, circuit, qubits):
        # Equation 4.8:
        for r, x in enumerate(self.X):
            for j in range(self.r, self.n-self.k):
                if x[j] == 'X' or x[j] == 'Y':
                    circuit.append(cirq.ControlledOperation([qubits[self.n - self.k + r]], cirq.X(qubits[j])))

        gate_dict = {'X': cirq.X, 'Y': cirq.Y, 'Z': cirq.Z}

        for r in range(self.r):
            circuit.append(cirq.H(qubits[r]))

            if self.M[r][r] == 'Y' or self.M[r][r] == 'Z':
                circuit.append(cirq.S(qubits[r]))

            for n in range(self.n):
                if n == r:
                    continue
                if self.M[r][n] == 'I':
                    continue
                op = gate_dict[self.M[r][n]]
                circuit.append(cirq.ControlledOperation([qubits[r]], op(qubits[n])))
        # At this stage, the state vector should be equal to equations 3.17 and 3.18.

    def compute_syndromes(self, circuit, qubits):
        qubit_map = {qubit: i for i, qubit in enumerate(qubits)}

        results = cirq.Simulator().simulate(circuit, qubit_order=qubits, initial_state=0)
        state_vector = results.state_vector()

        syndromes = []
        for r in range(self.r):
            pauli_string = cirq.PauliString(dict(zip(qubits, self.M[r])))
            trace = pauli_string.expectation_from_state_vector(state_vector, qubit_map)
            syndromes.append(round(trace.real))
        return tuple(syndromes)

    def correct(self, circuit, qubits, ancillas):
        # We set the ancillas so that measuring them directly would be the same
        # as measuring the qubits with Pauli strings. In other words, we store
        # the syndrome inside the ancillas.

        gate_dict = {'X': cirq.X, 'Y': cirq.Y, 'Z': cirq.Z}

        for r in range(self.r):
            circuit.append(cirq.H(ancillas[r]))
            for n in range(self.n):
                if self.M[r][n] == 'I':
                    continue
                op = gate_dict[self.M[r][n]]
                circuit.append(cirq.ControlledOperation([ancillas[r]], op(qubits[n])))
            circuit.append(cirq.H(ancillas[r]))

        # At this stage, the ancillas are equal to the syndrome. Now, we apply
        # the errors back to correct the code.

        for syndrome, correction in self.syndromes_to_corrections.items():
            op = gate_dict[correction[0]]
            n = correction[1]

            # We do a Boolean operation on the ancillas (i.e. syndrome).
            for r in range(self.r):
                if syndrome[r] == 1:
                    circuit.append(cirq.X(ancillas[r]))

            circuit.append(cirq.ControlledOperation(ancillas, op(qubits[n])))

            for r in range(self.r):
                if syndrome[r] == 1:
                    circuit.append(cirq.X(ancillas[r]))


code = StabilitizerCode(group_generators=['XZZXI', 'IXZZX', 'XIXZZ', 'ZXIXZ'])

print(code.syndromes_to_corrections)

circuit = cirq.Circuit()
qubits = [cirq.NamedQubit(name) for name in ['0', '1', '2', '3', 'c']]
ancillas = [cirq.NamedQubit(name) for name in ['d0', 'd1', 'd2', 'd3']]

code.encode(circuit, qubits)
circuit.append(cirq.X(qubits[3]))

code.correct(circuit, qubits, ancillas)

# for r in range(4):
#     circuit.append(cirq.measure(ancillas[r]))

#print(circuit)


results = cirq.Simulator().simulate(circuit, qubit_order=(qubits + ancillas), initial_state=1 * 2**4)
#print([1 - 2 * m.item() for m in results.measurements.values()])

state_vector = results.state_vector()
print(cirq.dirac_notation(4 * state_vector))


