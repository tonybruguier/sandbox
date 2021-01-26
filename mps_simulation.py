import cirq
import cirq.contrib.quimb as ccq
import resource
import sys
import traceback

soft, hard = resource.getrlimit(resource.RLIMIT_AS)
#resource.setrlimit(resource.RLIMIT_AS, (1 * 2 ** 30, soft))
resource.setrlimit(resource.RLIMIT_AS, (1 * 2 ** 30, hard))

# sys.stdout = open('stdout.txt', 'w')
# sys.stderr = open('stderr.txt', 'w')

try:
    # Load the circuit
    from n12_m14 import circuit_n12_m14_s0_e0_pEFGH
    circuit = circuit_n12_m14_s0_e0_pEFGH.CIRCUIT
    qubit_order = circuit_n12_m14_s0_e0_pEFGH.QUBIT_ORDER
    initial_state = 0

    # # Dense simulation
    # dense_simulator = cirq.Simulator()
    # dense_final = dense_simulator.simulate(circuit, qubit_order=qubit_order, initial_state=initial_state)
    # final_state_vector = dense_final.final_state_vector

    # MPS simulation
    mps_simulator = ccq.mps_simulator.MPSSimulator()
    mps_final = mps_simulator.simulate(circuit, qubit_order=qubit_order, initial_state=initial_state)
    final_state_vector = mps_finalactual.final_state.to_numpy()
except MemoryError as error:
    print('TONYBOOM caught memory error: %s' % (error))
    traceback.print_exc()
    sys.exit(1)
