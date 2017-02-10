import numpy as np

# converts cartesian index to index for the different observations
def cartesian_to_states(i, nbr_states, nbr_chains):

    vals = np.zeros((nbr_chains,), dtype=int)
    remainder = i
    for j in range(0, nbr_chains):
        temp = remainder % (nbr_states**(j+1))
        vals[nbr_chains-j-1] = temp / (nbr_states**j)
        remainder -= temp

    return vals

# converts indices for the observations to cartesian indices
def states_to_cartesian(states, nbr_states):

    i = 0
    nbr_chains = len(states)
    for j in range(0, nbr_chains):
        i += nbr_states**(nbr_chains-j-1)*states[j]

    return i

# returns the cartesian indices which has any element in common with any
# of the observation indices in states
def cartesian_indices_for(states, nbr_states):

    nbr_chains = len(states)

    ind_vectors = []
    for c in range(0, nbr_chains):
        d = nbr_chains - 1 - c # 2 for c = 0
        block = states[c]*nbr_states**d+np.arange(0, nbr_states**d)
        repeats = np.arange(0, nbr_states**nbr_chains, nbr_states**(d+1))
        I = (block[None, :] + repeats[:, None]).ravel()
        ind_vectors.append(I)

    return ind_vectors

# returns the indices which are overlapping in between any of the observation indices
def overlapping_indices_for(states, nbr_states):

    nbr_chains = len(states)

    ind_vectors = []
    for c in range(0, nbr_chains):
        d = nbr_chains - 1 - c # 2 for c = 0
        block = states[c]*nbr_states**d+np.arange(0, nbr_states**d, dtype=int)
        # get the overlap with the larger blocks
        I_vector = []
        for cc in range(0, c):
            dd = nbr_chains - 1 - cc
            inner_repeats = np.arange(0, nbr_states**dd, nbr_states**(d+1), dtype=int)
            offset_block = (block[None, :] + inner_repeats[:, None]).ravel()
            offset_block = offset_block + states[cc]*nbr_states**dd
            outer_repeats = np.arange(0, nbr_states**nbr_chains, nbr_states**(dd+1), dtype=int)
            I = (offset_block[None, :] + outer_repeats[:, None]).ravel()
            I_vector.append(I)

        ind_vectors.append(I_vector)

    return ind_vectors

def overlapping_indices_for_state(j, nbr_states, nbr_chains):

    states = j*np.ones((nbr_chains,), dtype=int)

    ind_vectors = overlapping_indices_for(states, nbr_states)
    print ind_vectors
    inds = np.concatenate([np.concatenate(I) for I in ind_vectors if len(I) > 0])

    return inds

# returns the Kronecker product for the row i, To_s is a generating parameter
def cartesian_transition_for_row(To_s, i, nbr_states, nbr_chains):
    states = cartesian_to_states(i, nbr_states, nbr_chains)
    if To_s == 1:
        To_s -= 0.001

    To_others = np.log(1.-To_s) - np.log(nbr_states)
    To_s = np.log(To_s)

    #T = -alpha*(t - last_t)*np.identity(nbr_states)
    #for i in range(0, T.shape[0]):
    #    mask = T[i, :] == 0.
    #    T[i, mask] = np.log(1.-np.exp(T[i, i])) - np.log(np.sum(mask))

    Ti = np.ones((1, 1))
    for c in range(0, nbr_chains):
        transitions = To_others*np.ones((nbr_states, 1))
        transitions[states[c], 0] = To_s
        #Ti = np.kron(Ti, transitions)
        Ti = (Ti[None, :] + transitions[:, None]).ravel().reshape(-1, 1)

    return Ti
