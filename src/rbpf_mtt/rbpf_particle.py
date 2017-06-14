import numpy as np
import math
import copy
from cartesian import cartesian_inds

def gauss_pdf(y, m, P):
    d = y - m
    #print y
    dim = len(y)
    det = np.linalg.det(P)
    if det < 0.0000001:
        return 0.
    denom = math.sqrt((2.*math.pi)**dim*np.linalg.det(P))
    if denom < 0.00001:
        print "DENOM TOO SMALL!", denom
        return 0.
    return 1./denom*math.exp(-0.5*np.dot(d, np.linalg.solve(P, d)))

def gauss_expected_likelihood(m, P):
    dim = len(m)
    det = np.linalg.det(P)
    if det < 0.0000001:
        print "DET TOO SMALL!", dim
        return 0.
    denom = 2.**dim*math.sqrt((math.pi)**dim*np.linalg.det(P))
    return 1./denom

# this is in principle just a Kalman filter over all the states
class RBPFMParticle(object):

    def __init__(self, spatial_dim, feature_dim, nbr_targets, nbr_locations, spatial_std, spatial_process_std,
                 feature_std, pjump=0.025, pnone=0.25, qjump=0.025, qnone=0.25):

        # Parameters
        self.nbr_locations = nbr_locations
        self.spatial_std = spatial_std
        self.spatial_process_std = spatial_process_std
        self.feature_std = feature_std
        self.fR = self.feature_std*self.feature_std*np.identity(feature_dim)
        self.pjump = pjump
        self.pnone = pnone
        self.location_area = 20.
        self.use_gibbs = False

        # State
        self.c = [] # the last assocations, can be dropped when new time arrives
        self.sm = np.zeros((nbr_targets, spatial_dim)) # the Kalman filter means
        self.fm = np.zeros((nbr_targets, feature_dim))
        self.sP = np.zeros((nbr_targets, spatial_dim, spatial_dim)) # the Kalman filter covariances
        self.fP = np.zeros((nbr_targets, feature_dim, feature_dim))
        self.location_ids = np.zeros((nbr_targets), dtype=int) # assigns targets to measurement sets
        self.location_unknown = np.zeros((nbr_targets), dtype=bool)
        self.last_time = -1

        # Debug: these properties are for inspection and non-essential to the algorithm
        self.did_jump = False
        self.nbr_jumps = 0
        self.nbr_noise = 0
        self.nbr_assoc = 0
        self.target_jumps = np.zeros((nbr_targets,))
        self.max_likelihoods = np.zeros((nbr_targets,))
        self.max_exp_likelihoods = np.zeros((nbr_targets,))
        self.sampled_modes = np.zeros((5,))

    def set_feature_cov(self, feature_cov):
        self.fR = feature_cov

    def predict(self, location_ids):

        spatial_process_std = self.spatial_process_std #0.25
        feature_process_std = 0.0 #1#1

        sQ = spatial_process_std*spatial_process_std*np.identity(self.sm.shape[1]) # process noise
        fQ = feature_process_std*feature_process_std*np.identity(self.fm.shape[1]) # process noise

        for j in range(0, self.sm.shape[0]):
            #self.sm[j] = self.sm[j]
            #self.fm[j] = self.fm[j]
            if np.sum(location_ids == self.location_ids[j]) > 0:
                self.sP[j] = self.sP[j] + sQ
                self.fP[j] = self.fP[j] + fQ

    def compute_prior(self, j, pjump, pnone, location_ids, nbr_observations):

        pprop = 1. - pjump # probability of local movement to this measurement
        pmeas = 1. - pnone
        pthis = 1./self.nbr_locations
        pother = 1. - pthis

        # 1. measure current room 2. measure jump current room 3. propagate no meas 4. jump no meas current room 5. jump to other room, no meas
        T = np.array([[pprop*pmeas, pthis*pjump*pmeas, pprop*pnone, pthis*pjump*pnone, pother*pjump],  
                      [0.,          pthis*pjump*pmeas, pprop,       pthis*pjump*pnone, pother*pjump], # pprop*pnone = 1.0
                      [0.,          pthis*pmeas,       0.,          pthis*pnone,       pother]]) # pjump = 1.0

        #print np.sum(T, axis=1)

        if self.location_unknown[j]: # the object has jumped to an unknown location
            mode = 2
        elif np.sum(location_ids == self.location_ids[j]) == 0: # the estimate is in a different location
            mode = 1
        else: # the estimate is in the same location as the measurement
            mode = 0

        prior = np.zeros((2*nbr_observations+3,))
        prior[:nbr_observations] = T[mode, 0]/float(nbr_observations)
        prior[nbr_observations:2*nbr_observations] = T[mode, 1]/float(nbr_observations)
        prior[2*nbr_observations] = T[mode, 2]
        prior[2*nbr_observations+1] = T[mode, 3]
        prior[2*nbr_observations+2] = T[mode, 4]

        return prior


    def target_compute_update(self, spatial_measurements, feature_measurements, sR, fR, location_ids):

        nbr_targets = self.sm.shape[0]
        spatial_dim = self.sm.shape[1]
        feature_dim = self.fm.shape[1]
        nbr_observations = spatial_measurements.shape[0]

        # Can we have 4-dimensional matrix in numpy, YES WE CAN!
        pot_sm = np.zeros((nbr_targets, nbr_observations, spatial_dim)) # the Kalman filter means
        pot_fm = np.zeros((nbr_targets, nbr_observations, feature_dim))
        pot_sP = np.zeros((nbr_targets, nbr_observations, spatial_dim, spatial_dim)) # the Kalman filter covariances
        pot_fP = np.zeros((nbr_targets, nbr_observations, feature_dim, feature_dim))
        likelihoods = np.zeros((nbr_targets, 2*nbr_observations+3))
        weights = np.zeros((nbr_targets,))
        
        spatial_likelihoods = np.zeros((nbr_targets, nbr_observations))
        feature_likelihoods = np.zeros((nbr_targets, nbr_observations))
        spatial_expected_likelihoods = np.zeros((nbr_targets,))
        feature_expected_likelihoods = np.zeros((nbr_targets,))

        self.predict(location_ids)

        # compute the likelihoods for all the observations and targets
        for k in range(0, nbr_targets):

            sS = self.sP[k] + sR # IS = (R + H*P*H');
            fS = self.fP[k] + fR

            # sP^T = sP, sR^t = sR -> sS^-1*sP = sP*sS^-1
            sK = np.linalg.solve(sS, self.sP[k]) # K = P*H'/IS;
            fK = np.linalg.solve(fS, self.fP[k])

            for j in range(0, nbr_observations):

                sy = spatial_measurements[j] - self.sm[k]
                fy = feature_measurements[j] - self.fm[k]

                pot_sm[k, j] = self.sm[k] + np.dot(sK, sy) # X = X + K * (y-IM);
                pot_fm[k, j] = self.fm[k] + np.dot(fK, fy)
                pot_sP[k, j] = np.dot((np.identity(spatial_dim) - sK), self.sP[k])
                pot_fP[k, j] = np.dot((np.identity(feature_dim) - fK), self.fP[k])

                # TODO: wait 1s, didn't they write that this should be pot_sm[j]?
                spatial_likelihoods[k, j] = gauss_pdf(spatial_measurements[j], self.sm[k], sS)
                feature_likelihoods[k, j] = gauss_pdf(feature_measurements[j], self.fm[k], fS)
        
            spatial_expected_likelihoods[k] = gauss_expected_likelihood(self.sm[k], sS)
            feature_expected_likelihoods[k] = gauss_expected_likelihood(self.fm[k], fS)

        # Compute all the likelihoods
        for k in range(0, nbr_targets):
            
            prior = self.compute_prior(k, self.pjump, self.pnone, location_ids, nbr_observations)
            
            likelihood = np.zeros((2*nbr_observations+3,))

            likelihood[:nbr_observations] = spatial_likelihoods[k, :]*feature_likelihoods[k, :]
            likelihood[nbr_observations:2*nbr_observations] = (1./self.location_area)*feature_likelihoods[k, :]
            likelihood[2*nbr_observations:] = 1./float(nbr_targets)*spatial_expected_likelihoods[k]*feature_expected_likelihoods[k]
            
            proposal = prior*likelihood

            weights[k] = np.sum(proposal)
            proposal *= 1./weights[k]

            likelihoods[k] = proposal

            self.max_likelihoods[k] = np.max(spatial_likelihoods[k, :]*feature_likelihoods[k, :])
            self.max_exp_likelihoods[k] = 1./float(nbr_targets)*spatial_expected_likelihoods[k]*feature_expected_likelihoods[k]

        return likelihoods, weights, pot_sm, pot_fm, pot_sP, pot_fP


    def target_sample_update(self, nbr_observations, likelihoods, location_ids):

        nbr_targets = self.sm.shape[0]
        spatial_dim = self.sm.shape[1]
        feature_dim = self.fm.shape[1]

        # sample measurement -> target mapping
        states = np.zeros((nbr_targets,), dtype=int)
        sampled_states = np.zeros((nbr_targets,), dtype=int)
        while True:
            sampled_states = -1*np.ones((nbr_targets,), dtype=int)
            nbr_no_meas = 0
            for j in range(0, nbr_targets):
                # the problem here is that it does not take the other probabilites into account
                states[j] = np.random.choice(2*nbr_observations+3, p=likelihoods[j])
                if states[j] < 2*nbr_observations:
                    sampled_states[j] = states[j] % nbr_observations
                else:
                    nbr_no_meas += 1

            unique, counts = np.unique(sampled_states, return_counts=True)
            nbr_sampled = len(unique) - int(nbr_no_meas > 0)

            if nbr_sampled + nbr_no_meas == nbr_targets:
                #unique_list = unique.tolist()
                nbr_jumps = np.sum(states >= nbr_observations) - nbr_no_meas
                nbr_assoc = np.sum(states < nbr_observations)
                #print "Found assoc: ", nbr_assoc, ", nbr jumps: ", nbr_jumps, ", no meas: ", nbr_no_meas
                break

        for j in range(0, nbr_targets):

            self.c.append(sampled_states[j])
            #self.location_unknown[j] = False # associated with target or jump
            if states[j] == 2*nbr_observations+2:
                self.location_unknown[j] = True
            elif self.location_unknown[j]:
                self.location_unknown[j] = False
                self.location_ids[j] = location_ids[0]
            elif sampled_states[j] > 0:
                self.location_ids[j] = location_ids[sampled_states[j]]
            #else:
            #    self.location_ids[j] = location_ids[0]

        return states
    
    def gibbs_sample_update(self, nbr_observations, likelihoods, weights, location_ids):

        nbr_targets = self.sm.shape[0]
        spatial_dim = self.sm.shape[1]
        feature_dim = self.fm.shape[1]
        
        states = np.zeros((nbr_targets,), dtype=int)
        while True:
            for j in range(0, nbr_targets):
                states[j] = np.random.choice(2*nbr_observations+3, 1, p=likelihoods[j])
            b = states[states < 2*nbr_observations]
            if len(np.unique(np.mod(b, nbr_observations))) == len(b):
                break
        
        for n in range(0, 1000):
            inds = np.random.choice(nbr_targets, 2, replace=False)
            nbr_assigned = np.sum(states < 2*nbr_observations)
            marginals1 = np.array(weights[0]*likelihoods[inds[0]])
            marginals2 = np.array(weights[1]*likelihoods[inds[1]])
            # find all inds where a < nbr_observations and != inds[1] or inds[0]
            for j, v in enumerate(states.tolist()):
                if j != inds[0] and j != inds[1] and v < 2*nbr_observations:
                    marginals1[v%nbr_observations+nbr_observations] = 0.
                    marginals2[v%nbr_observations+nbr_observations] = 0.
                    marginals1[v%nbr_observations] = 0.
                    marginals2[v%nbr_observations] = 0.
            if nbr_observations > nbr_assigned:
                marginals1[:2*nbr_observations] *= float(nbr_observations)/float(nbr_observations-nbr_assigned)
                marginals2[:2*nbr_observations] *= float(nbr_observations)/float(nbr_observations-nbr_assigned)

            marginals1 *= 1./np.sum(marginals1)
            marginals2 *= 1./np.sum(marginals2) 
            joint_pair = np.kron(marginals1, marginals2)
            inds1 = np.kron(np.arange(0, 2*nbr_observations+3), np.ones((2*nbr_observations+3,)))
            inds2 = np.kron(np.ones((2*nbr_observations+3,)), np.arange(0, 2*nbr_observations+3))
            zinds = np.logical_and(np.logical_and(np.mod(inds1, nbr_observations) == np.mod(inds2, nbr_observations),
                                   inds1 < 2*nbr_observations), inds2 < 2*nbr_observations)
            joint_pair[zinds] = 0
            joint_pair *= 1./np.sum(joint_pair)
            state_pair = np.random.choice((2*nbr_observations+3)**2, 1, p=joint_pair)
            state1 = inds1[state_pair]
            state2 = inds2[state_pair]

            states[inds[0]] = state1
            states[inds[1]] = state2

        sampled_states = np.zeros((nbr_targets,), dtype=int)
        sampled_states[states < 2*nbr_observations] = np.mod(states[states < 2*nbr_observations], nbr_observations)
        sampled_states[states >= 2*nbr_observations] = -1

        for j in range(0, nbr_targets):

            self.c.append(sampled_states[j])
            #self.location_unknown[j] = False # associated with target or jump
            if states[j] == 2*nbr_observations+2:
                self.location_unknown[j] = True
            elif self.location_unknown[j]:
                self.location_unknown[j] = False
                self.location_ids[j] = location_ids[0]
            elif sampled_states[j] > 0:
                self.location_ids[j] = location_ids[sampled_states[j]]
            #else:
            #    self.location_ids[j] = location_ids[0]

        return states

    # this functions takes in several measurements at the same timestep
    # and does association jointly, leading to fewer particles with low weights
    def update(self, spatial_measurements, feature_measurements, time, observation_id, location_ids):

        self.c = []
        self.did_jump = False

        if time != self.last_time:
            self.c = []
            self.last_time = time

        #print "Got measurement with observations: ", spatial_measurements.shape[0]

        spatial_var = self.spatial_std*self.spatial_std
        feature_var = self.feature_std*self.feature_std

        nbr_targets = self.sm.shape[0]
        spatial_dim = self.sm.shape[1]
        feature_dim = self.fm.shape[1]
        nbr_observations = spatial_measurements.shape[0]

        sR = spatial_var*np.identity(spatial_dim) # measurement noise
        fR = self.fR

        likelihoods, weights, pot_sm, pot_fm, pot_sP, pot_fP = \
            self.target_compute_update(spatial_measurements, feature_measurements, sR, fR, location_ids)

        if self.use_gibbs:
            states = self.gibbs_sample_update(nbr_observations, likelihoods, weights, location_ids)
        else:
            states = self.target_sample_update(nbr_observations, likelihoods, location_ids)

        weight_update = np.prod(weights)

        for k, i in enumerate(states):

            self.did_jump = False
            if i == 2*nbr_observations+2:
                self.nbr_noise += 1
                self.sampled_modes[4] += 1
            elif i == 2*nbr_observations+1:
                self.nbr_noise += 1
                self.nbr_jumps += 1
                self.did_jump = True
                self.sampled_modes[3] += 1
            elif i == 2*nbr_observations:
                self.nbr_noise += 1
                self.sampled_modes[2] += 1
            elif i >= nbr_observations:
                self.sm[k] = spatial_measurements[i % nbr_observations]
                self.sP[k] = spatial_var*np.eye(spatial_dim)
                self.did_jump = True
                self.nbr_jumps += 1
                self.target_jumps[k] += 1
                self.sampled_modes[1] += 1
            else:
                self.sm[k] = pot_sm[k, i]
                self.fm[k] = pot_fm[k, i]
                self.sP[k] = pot_sP[k, i]
                self.fP[k] = pot_fP[k, i]
                self.nbr_assoc += 1
                self.sampled_modes[0] += 1

        return weight_update
