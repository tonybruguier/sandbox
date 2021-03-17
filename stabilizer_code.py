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

    def encode(self, circuit, qubits):
        gate_dict = {'X': cirq.X, 'Y': cirq.Y, 'Z': cirq.Z}

        # Equation 4.8:
        for i, x in enumerate(self.X):
            for j in range(self.r, self.n-self.k):
                if x[j] == 'X' or x[j] == 'Y':
                    circuit.append(cirq.ControlledOperation(
                        [qubits[self.n - self.k + i]], cirq.X(qubits[j])))

        for i in range(self.r):
            circuit.append(cirq.H(qubits[i]))

            if self.M[i][i] == 'Y' or self.M[i][i] == 'Z':
                circuit.append(cirq.S(qubits[i]))

            for j in range(self.n):
                if j == i:
                    continue
                if self.M[i][j] == 'I':
                    continue
                op = gate_dict[self.M[i][j]]
                circuit.append(cirq.ControlledOperation([qubits[i]], op(qubits[j])))

        # At this stage, the state vector should be equal to equations 3.17 and 3.18.

def test_decoder(input_val, error_loc=None):
    code = StabilitizerCode(group_generators=['XZZXI', 'IXZZX', 'XIXZZ', 'ZXIXZ'])

    circuit = cirq.Circuit()
    qubits = [cirq.NamedQubit(name) for name in ['0', '1', '2', '3', 'c', 'd']]
    qubit_map = {qubit: i for i, qubit in enumerate(qubits)}

    code.encode(circuit, qubits)

    if error_loc:
        circuit.append(cirq.X(qubits[error_loc]))

    for r in range(code.k):
        for i in range(code.n):
            if code.Z[r][i] == 'Z':
                circuit.append(cirq.ControlledOperation(
                    [qubits[i]], cirq.X(qubits[code.n + r])))



    results = cirq.Simulator().simulate(circuit, qubit_order=qubits, initial_state=(input_val * 2))
    state_vector = results.state_vector()

    # print(circuit)
    # print(cirq.dirac_notation(state_vector * 4))

    pauli_string = cirq.PauliString(dict(zip(qubits, 'IIIIIZ')))
    trace = pauli_string.expectation_from_state_vector(state_vector, qubit_map)

    return trace

for error_loc in [None, 0, 1, 2, 3, 4]:
    for input_val in range(2):
# for error_loc in [None]:
#     for input_val in range(1, 2):
        print('\n')
        for _ in range(1):
            decoded_val = test_decoder(input_val, error_loc)
            print('error_loc=%s\tinput=%d\tdecoded=%s' % (error_loc, input_val, decoded_val))
