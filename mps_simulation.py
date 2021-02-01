import cirq
import cirq.contrib.quimb as ccq
import numpy as np
import resource
import sys
import traceback

soft, hard = resource.getrlimit(resource.RLIMIT_AS)
#resource.setrlimit(resource.RLIMIT_AS, (1 * 2 ** 30, soft))
resource.setrlimit(resource.RLIMIT_AS, ((1024 + 512) * 2 ** 20, hard))


try:
    # Load the circuit
    from n12_m14 import circuit_n12_m14_s0_e0_pEFGH
    circuit = circuit_n12_m14_s0_e0_pEFGH.CIRCUIT
    qubit_order = circuit_n12_m14_s0_e0_pEFGH.QUBIT_ORDER
    initial_state = 0

    # Dense simulation
    dense_simulator = cirq.Simulator()
    dense_final = dense_simulator.simulate(circuit, qubit_order=qubit_order, initial_state=initial_state)
    dense_final_state_vector = dense_final.final_state_vector

    # MPS simulation
    grouping = {qubit: qubit.col for qubit in qubit_order}

    mps_simulator = ccq.mps_simulator.MPSSimulator(rsum2_cutoff=1e-2, grouping=grouping)
    mps_final = mps_simulator.simulate(circuit, qubit_order=qubit_order, initial_state=initial_state)
    mps_final_state_vector = mps_final.final_state.to_numpy()

    fidelity = np.einsum('i,j,j,i->',
        dense_final_state_vector.conj(),
        dense_final_state_vector,
        mps_final_state_vector.conj(),
        mps_final_state_vector)

    print('%s' % (mps_final.final_state.estimation_stats()))

    print('fidelity=%.3e' % (fidelity.real))

except MemoryError as error:
    print('TONYBOOM caught memory error: %s' % (error))
    traceback.print_exc()
    sys.exit(1)
