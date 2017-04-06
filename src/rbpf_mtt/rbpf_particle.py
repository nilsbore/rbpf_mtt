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

    def __init__(self, spatial_dim, feature_dim, nbr_targets, spatial_std,
                 feature_std, pjump=0.025, pnone=0.25, qjump=0.025, qnone=0.25):

        self.spatial_std = spatial_std
        self.feature_std = feature_std
        self.fR = self.feature_std*self.feature_std*np.identity(feature_dim)

        self.c = [] # the last assocations, can be dropped when new time arrives
        self.sm = np.zeros((nbr_targets, spatial_dim)) # the Kalman filter means
        self.fm = np.zeros((nbr_targets, feature_dim))
        self.sP = np.zeros((nbr_targets, spatial_dim, spatial_dim)) # the Kalman filter covariances
        self.fP = np.zeros((nbr_targets, feature_dim, feature_dim))
        self.measurement_partitions = np.zeros((nbr_targets), dtype=int) # assigns targets to measurement sets
        self.location_ids = np.zeros((nbr_targets), dtype=int) # assigns targets to measurement sets
        self.might_have_jumped = np.zeros((nbr_targets), dtype=bool)
        self.last_time = -1
        self.associations = {}
        self.did_jump = False
        self.nbr_jumps = 0
        self.nbr_noise = 0
        self.nbr_assoc = 0
        self.target_jumps = np.zeros((nbr_targets,))

        self.pjump = pjump
        self.pnone = pnone

        self.qjump = qjump
        self.qnone = qnone

        #self.current_
        # the way this is supposed to work is that, when we sample a new c, we can only do it within one set

    def set_feature_cov(self, feature_cov):
        self.fR = feature_cov

    def predict(self, location_ids, measurement_partition=None):

        spatial_process_std = 0.25
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

        pprop = 1.0 - pjump # probability of local movement to this measurement

        if np.sum(location_ids == self.location_ids[j]) == 0: # no observations from target room
            if self.might_have_jumped[j]:
                pnone = 1. - pjump*(1. - pnone)*((1.-self.pjump)*(1. - self.pnone))/self.pnone # computed so that the jump probabilities stay the same!
            else:
                pnone = 1. - pjump*(1. - pnone) # computed so that the jump probabilities stay the same!
            pprop = 0. # can't have any local movement here!

        prior = np.zeros((2*nbr_observations+1))
        prior_norm = (1. - pnone) / (pprop + pjump) / float(nbr_observations)
        prior[:nbr_observations] = pprop * prior_norm
        prior[nbr_observations:2*nbr_observations] = pjump * prior_norm
        prior[2*nbr_observations] = pnone

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
        likelihoods = np.zeros((nbr_targets, 2*nbr_observations+1))
        prop_ratios = np.zeros((nbr_targets, 2*nbr_observations+1))
        weights = np.zeros((nbr_targets,))
        #pc = np.zeros((nbr_targets, 2*nbr_observations+1))

        spatial_likelihoods = np.zeros((nbr_targets, nbr_observations))
        feature_likelihoods = np.zeros((nbr_targets, nbr_observations))

        self.predict(location_ids)

        # compute the likelihoods for all the observations and targets
        for k in range(0, nbr_targets):

            prior = self.compute_prior(k, self.pjump, self.pnone, location_ids, nbr_observations)
            prop_prior = self.compute_prior(k,self.qjump, self.qnone, location_ids, nbr_observations)

            likelihood = np.zeros((2*nbr_observations+1))

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

            #likelihoods[k, :nbr_observations] = spatial_likelihoods[k, :]*feature_likelihoods[k, :]
            #likelihoods[k, nbr_observations:2*nbr_observations] = pjump*feature_likelihoods[k, :]
            #likelihoods[k, 2*nbr_observations] = target_pnone
            #weights[k] = np.sum(likelihoods[k])
            #likelihoods[k] = 1./weights[k]*likelihoods[k]

            spatial_expected_likelihood = gauss_expected_likelihood(self.sm[k], sS)
            feature_expected_likelihood = gauss_expected_likelihood(self.fm[k], fS)

            likelihood[:nbr_observations] = spatial_likelihoods[k, :]*feature_likelihoods[k, :]
            likelihood[nbr_observations:2*nbr_observations] = spatial_expected_likelihood*feature_likelihoods[k, :]
            likelihood[2*nbr_observations] = spatial_expected_likelihood*feature_expected_likelihood
            prop_proposal = prop_prior*likelihood
            proposal = prior*likelihood

            weight = np.sum(proposal)
            prop_proposal *= 1./np.sum(prop_proposal)
            proposal *= 1./weight

            weights[k] = weight
            likelihoods[k] = prop_proposal

            prop_proposal[prop_proposal == 0] = 1.
            prop_ratios[k] = proposal / prop_proposal

        return likelihoods, weights, prop_ratios, pot_sm, pot_fm, pot_sP, pot_fP


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
                states[j] = np.random.choice(2*nbr_observations+1, p=likelihoods[j])
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

            # observation from target room but associated with noise
            if np.sum(location_ids == self.location_ids[j]) > 0 and sampled_states[j] == -1:
                self.might_have_jumped[j] = True

            self.c.append(sampled_states[j])
            if sampled_states[j] > 0:
                self.might_have_jumped[j] = False # associated with target or jump
                self.location_ids[j] = location_ids[sampled_states[j]]
            #self.last_c[j] = sampled_states[j]

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
        fR = self.fR #feature_var*np.identity(feature_dim) # measurement noise

        likelihoods, weights, prop_ratios, pot_sm, pot_fm, pot_sP, pot_fP = \
            self.target_compute_update(spatial_measurements, feature_measurements, sR, fR, location_ids)

        states = self.target_sample_update(nbr_observations, likelihoods, location_ids)

        weight_update = np.prod(weights)

        for k, i in enumerate(states):

            weight_update *= prop_ratios[k, i]

            if i == 2*nbr_observations:
                self.nbr_noise += 1
                #self.c.append(nbr_targets) # to know that we observed noise!
                #weights_update *= 0.1 #1.0#likelihoods[k, i]/pc[k, i]
            elif i >= nbr_observations:
                self.sm[k] = spatial_measurements[i % nbr_observations]
                self.sP[k] = spatial_var*np.eye(spatial_dim)
                self.did_jump = True
                self.nbr_jumps += 1
                self.target_jumps[k] += 1
                #weights_update *= 0.2
            else:
                self.sm[k] = pot_sm[k, i]
                self.fm[k] = pot_fm[k, i]
                self.sP[k] = pot_sP[k, i]
                self.fP[k] = pot_fP[k, i]
                self.associations[observation_id] = i
                self.nbr_assoc += 1
                #weights_update *= 1.0#likelihoods[k, i]/pc[k, i]

        return weight_update
