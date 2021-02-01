import cirq
import cirq.contrib.quimb as ccq
import numpy as np
import resource
import sys
import traceback

_, hard = resource.getrlimit(resource.RLIMIT_AS)
resource.setrlimit(resource.RLIMIT_AS, ((1024 + 512) * 2 ** 20, hard))

try:
    # Load the circuit
    from n40_m14 import circuit_n12_m14_s0_e0_pEFGH as example
    circuit = example.CIRCUIT
    qubit_order = example.QUBIT_ORDER
    initial_state = 0

    # MPS simulation
    min_diag = min([qubit.col - qubit.row for qubit in qubit_order])
    grouping = {qubit: (qubit.col - qubit.row - min_diag) for qubit in qubit_order}

    mps_simulator = ccq.mps_simulator.MPSSimulator(rsum2_cutoff=0.1, grouping=grouping)
    mps_final = mps_simulator.simulate(circuit, qubit_order=qubit_order, initial_state=initial_state)

    print('%s' % (mps_final.final_state.estimation_stats()))

    # Dense simulation
    mps_final_state_vector = mps_final.final_state.to_numpy()

    dense_simulator = cirq.Simulator()
    dense_final = dense_simulator.simulate(circuit, qubit_order=qubit_order, initial_state=initial_state)
    dense_final_state_vector = dense_final.final_state_vector

    fidelity = np.einsum('i,j,j,i->',
        dense_final_state_vector.conj(),
        dense_final_state_vector,
        mps_final_state_vector.conj(),
        mps_final_state_vector)

    print('fidelity=%.3e' % (fidelity.real))

except MemoryError as error:
    print('TONYBOOM caught memory error: %s' % (error))
    traceback.print_exc()
    sys.exit(1)
