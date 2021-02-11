import cirq
import cirq.contrib.quimb as ccq
import numpy as np
import resource
import sys
import traceback

# _, hard = resource.getrlimit(resource.RLIMIT_AS)
# resource.setrlimit(resource.RLIMIT_AS, (1 * 2 ** 30, hard))

def pretty_print_grouping(qubit_order, grouping):
    max_col = max([qubit.col for qubit in qubit_order])
    max_row = max([qubit.row for qubit in qubit_order])
    group_strings = []
    for row in range(0, max_row + 1):
        group_strings.append(['  '] * (max_col + 1))

    for qubit, group in grouping.items():
        group_strings[qubit.row][qubit.col] = '%2d' % (group)

    group_strings = '\n\n'.join([' '.join(row) for row in group_strings])

    print('Grouping:\n%s' % (group_strings))

def assign_grouping(qubit_order):

    grouping = {qubit: [qubit.row, qubit.col] for qubit in qubit_order}
    # grouping = {qubit: [qubit.col - qubit.row, qubit.col + qubit.row] for qubit in qubit_order}

    min_row = min([point[0] for point in grouping.values()])
    min_col = min([point[1] for point in grouping.values()])
    for point in grouping.values():
        point[0] -= min_row
        point[1] -= min_col

    for point in grouping.values():
        point[0] //= 1
        point[1] //= 1

    max_row = max([point[0] for point in grouping.values()])
    for qubit in grouping.keys():
        grouping[qubit] = grouping[qubit][0] + grouping[qubit][1] * max_row
        # grouping[qubit] = grouping[qubit][1]

    return grouping

try:
    # Load the circuit
    from n12_m14 import circuit_n12_m14_s0_e0_pEFGH as example
    # from n36_m14 import circuit_n36_m14_s0_e0_pEFGH as example
    # from n53_m20 import circuit_n53_m20_s0_e0_pABCDCDAB as example
    circuit = example.CIRCUIT
    qubit_order = example.QUBIT_ORDER
    initial_state = 0

    # Grouping
    grouping = assign_grouping(qubit_order)
    pretty_print_grouping(qubit_order, grouping)

     # MPS simulation
    mps_simulator = ccq.mps_simulator.MPSSimulator(rsum2_cutoff=1e-3, grouping=grouping)
    mps_final = mps_simulator.simulate(circuit, qubit_order=qubit_order, initial_state=initial_state)

    print('%s' % (mps_final.final_state.estimation_stats()))

    # # Dense simulation
    # mps_final_state_vector = mps_final.final_state.to_numpy()

    # dense_simulator = cirq.Simulator()
    # dense_final = dense_simulator.simulate(circuit, qubit_order=qubit_order, initial_state=initial_state)
    # dense_final_state_vector = dense_final.final_state_vector

    # fidelity = np.einsum('i,j,j,i->',
    #     dense_final_state_vector.conj(),
    #     dense_final_state_vector,
    #     mps_final_state_vector.conj(),
    #     mps_final_state_vector)

    # print('fidelity=%.3e' % (fidelity.real))

except MemoryError as error:
    print('Caught memory error: %s' % (error))
    traceback.print_exc()
    sys.exit(1)
