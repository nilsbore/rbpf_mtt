import numpy as np
import math
import copy
from cartesian import cartesian_inds

def gauss_pdf(y, m, P):
    d = y - m
    denom = math.sqrt(2.*math.pi*np.linalg.det(P))
    if denom < 0.00001:
        return 0.
    return 1./denom*math.exp(-0.5*np.dot(d, np.linalg.solve(P, d)))

# this is in principle just a Kalman filter over all the states
class RBPFMParticle(object):

    def __init__(self, spatial_dim, feature_dim, nbr_targets):

        self.c = [] # the last assocations, can be dropped when new time arrives
        self.sm = np.zeros((nbr_targets, spatial_dim)) # the Kalman filter means
        self.fm = np.zeros((nbr_targets, feature_dim))
        self.sP = np.zeros((nbr_targets, spatial_dim, spatial_dim)) # the Kalman filter covariances
        self.fP = np.zeros((nbr_targets, feature_dim, feature_dim))
        self.measurement_partitions = np.zeros((nbr_targets), dtype=int) # assigns targets to measurement sets
        self.last_time = -1
        self.associations = {}
        # the way this is supposed to work is that, when we sample a new c, we can only do it within one set

    def predict(self, measurement_partition=None):

        spatial_process_noise = 0.2
        feature_process_noise = 0.05

        sQ = np.identity(self.sm.shape[1]) # process noise
        fQ = np.identity(self.fm.shape[1]) # process noise

        for j in range(0, self.sm.shape[0]):
            self.sm[j] = self.sm[j]
            self.fm[j] = self.fm[j]
            self.sP[j] = self.sP[j] + sQ
            self.fP[j] = self.fP[j] + fQ

    # this again highlights that we should take all measurements in one go
    def negative_update(self, spatial_measurement, observation_id):

        if observation_id not in self.associations:
            return 1.

        j = self.associations[observation_id]
        sR = spatial_measurement_noise*np.identity(spatial_dim)
        neg_likelihood = gauss_pdf(spatial_measurement, self.sm[j], sR)

        return 1. - neg_likelihood


    # spatial_measurement is a vector with just one measurement of the object position
    # feature_measurement is a vector with a measurement of the object feature
    # time is a unique identifier > 0 for the measurement occasion
    def update(self, spatial_measurement, feature_measurement, time, observation_id):

        #if len(self.c) == 0 or self.c[0] == 0:
        #    print "No associations sampled yet, can't update..."
        #    return

        if time != self.last_time:
            self.c = []
            self.last_time = time

        # we somehow need to integrate the feature density into these clutter things
        pclutter = 0.00002 # probability of measurement originating from noise
        pdclutter = 0.00002 # probability density of clutter measurement
        spatial_measurement_noise = 0.4
        feature_measurement_noise = 0.2
        pjump = 0.02

        # First find out the association
        # likelihoods for each target given
        # each hypotheses (particles). Store the
        # updated mean and covariance, and likelihood
        # for each association hypothesis.

        nbr_targets = self.sm.shape[0]
        spatial_dim = self.sm.shape[1]
        feature_dim = self.fm.shape[1]

        sR = spatial_measurement_noise*np.identity(spatial_dim) # measurement noise
        fR = feature_measurement_noise*np.identity(feature_dim) # measurement noise

        pot_sm = np.zeros((nbr_targets, spatial_dim)) # the Kalman filter means
        pot_fm = np.zeros((nbr_targets, feature_dim))
        pot_sP = np.zeros((nbr_targets, spatial_dim, spatial_dim)) # the Kalman filter covariances
        pot_fP = np.zeros((nbr_targets, feature_dim, feature_dim))
        likelihoods = np.zeros(nbr_targets+1,)

        spatial_likelihoods = np.zeros((nbr_targets,))
        feature_likelihoods = np.zeros((nbr_targets,))
        for j in range(0, nbr_targets):

            #
            # update step
            #
            #IM = H*X;
            #IS = (R + H*P*H');
            #K = P*H'/IS;
            #X = X + K * (y-IM);
            #P = P - K*IS*K';
            #if nargout > 5
            #   LH = gauss_pdf(y,IM,IS);
            #end

            # IM = H*X; - no process model!

            sy = spatial_measurement - self.sm[j]
            fy = feature_measurement - self.fm[j]

            sS = self.sP[j] + sR # IS = (R + H*P*H');
            fS = self.fP[j] + fR

            # sP^T = sP, sR^t = sR -> sS^-1*sP = sP*sS^-1
            sK = np.linalg.solve(sS, self.sP[j]) # K = P*H'/IS;
            fK = np.linalg.solve(fS, self.fP[j])

            #print sK.shape
            #print self.sm[j].shape
            #print (sK*sy).shape
            #print sy.shape

            pot_sm[j] = self.sm[j] + np.dot(sK, sy) # X = X + K * (y-IM);
            pot_fm[j] = self.fm[j] + np.dot(fK, fy)
            pot_sP[j] = np.dot((np.identity(spatial_dim) - sK), self.sP[j])
            pot_fP[j] = np.dot((np.identity(feature_dim) - fK), self.fP[j])

            # TODO: wait 1s, didn't they write that this should be pot_sm[j]?
            spatial_likelihoods[j] = gauss_pdf(spatial_measurement, self.sm[j], sS)
            feature_likelihoods[j] = gauss_pdf(feature_measurement, self.fm[j], fS)
            #likelihoods[j] = gauss_pdf(spatial_measurement, self.sm[j], sS) * \
            #                 gauss_pdf(feature_measurement, self.fm[j], fS)
        likelihoods[:nbr_targets] = spatial_likelihoods*feature_likelihoods
        likelihoods[nbr_targets] = pdclutter
        #likelihoods = 1./np.sum(likelihoods)*likelihoods

        # Optimal importance functions for target
        # and clutter associations (i) for each particles (j)
        #
        #     PC(i,j) = p(c[k]==i | c[k-1], y), c=T+1 means clutter.

        # INSERT CODE HERE

        # CP - Prior probability of a measurement being due
        # to clutter.

        # TP = [(1-repmat(CP,size(TP,1),1)).*TP;CP];
        # PC = LH .* TP;
        # sp = sum(PC,1);
        # ind = find(sp==0);
        # if ~isempty(ind)
        #   sp(ind)   = 1;
        #   PC(:,ind) = ones(size(PC,1),size(ind,2))/size(PC,1);
        # end
        # PC = PC ./ repmat(sp,size(PC,1),1);

        pc = np.zeros((nbr_targets+2,))
        # probability of measurement given association, may arise from clutter anyways?
        pc[:nbr_targets] = (1.-pclutter)*likelihoods[:nbr_targets]
        pc[nbr_targets] = pclutter*likelihoods[nbr_targets]
        pc[nbr_targets+1] = pjump
        for picked in self.c:
            pc[picked] = 0.
        pc = 1./np.sum(pc)*pc

        # Associate each particle to random target
        # or clutter using the importance distribution
        # above. Also calculate the new weights and
        # perform corresponding updates.

        # INSERT CODE HERE
        # TP - Tx1 vector of prior probabilities for measurements
        # hitting each of the targets. (optional, default uniform)
        # S{j}.W = S{j}.W * LH(i,j) * TP(i) / PC(i,j);

        i = np.random.choice(nbr_targets+2, p=pc) #categ_rnd()
        # so what happens here if i ==
        if i == nbr_targets+1:
            weights_update = pjump
        else:
            weights_update = likelihoods[i]/pc[i]

        if i == nbr_targets:
            pass
            #self.c = nbr_targets # measurement is noise
            #we don't really care if it was associated with noise

        elif i == nbr_targets+1:
            for picked in self.c:
                feature_likelihoods[picked] = 0.
            feature_likelihoods = feature_likelihoods/np.sum(feature_likelihoods)
            i = np.random.choice(nbr_targets, p=feature_likelihoods)
            self.sm[i] = spatial_measurement
            self.sP[i] = 1.0*np.eye(len(spatial_measurement))
        else:
            self.sm[i] = pot_sm[i]
            self.fm[i] = pot_fm[i]
            self.sP[i] = pot_sP[i]
            self.fP[i] = pot_fP[i]
            self.c.append(i)
            self.associations[observation_id] = i

        # Normalize the particles

        # Note: this should happen in the filter

        return weights_update

    # this functions takes in several measurements at the same timestep
    # and does association jointly, leading to fewer particles with low weights
    def joint_update(self, spatial_measurements, feature_measurements, time, observation_id):

        if time != self.last_time:
            self.c = []
            self.last_time = time

        print "Got measurement with observations: ", spatial_measurements.shape[0]

        nbr_observations = spatial_measurements.shape[0]

        # we somehow need to integrate the feature density into these clutter things
        pclutter = 0.00002 # probability of measurement originating from noise
        pdclutter = 0.00002 # probability density of clutter measurement
        spatial_measurement_noise = 0.4
        feature_measurement_noise = 0.2
        pjump = 0.002

        nbr_targets = self.sm.shape[0]
        spatial_dim = self.sm.shape[1]
        feature_dim = self.fm.shape[1]

        sR = spatial_measurement_noise*np.identity(spatial_dim) # measurement noise
        fR = feature_measurement_noise*np.identity(feature_dim) # measurement noise

        # Can we have 4-dimensional matrix in numpy, YES WE CAN!
        pot_sm = np.zeros((nbr_observations, nbr_targets, spatial_dim)) # the Kalman filter means
        pot_fm = np.zeros((nbr_observations, nbr_targets, feature_dim))
        pot_sP = np.zeros((nbr_observations, nbr_targets, spatial_dim, spatial_dim)) # the Kalman filter covariances
        pot_fP = np.zeros((nbr_observations, nbr_targets, feature_dim, feature_dim))
        likelihoods = np.zeros((nbr_observations, nbr_targets+1))
        pc = np.zeros((nbr_observations, nbr_targets+2))

        spatial_likelihoods = np.zeros((nbr_observations, nbr_targets))
        feature_likelihoods = np.zeros((nbr_observations, nbr_targets))

        for k in range(0, nbr_observations):
            for j in range(0, nbr_targets):

                sy = spatial_measurements[k] - self.sm[j]
                fy = feature_measurements[k] - self.fm[j]

                sS = self.sP[j] + sR # IS = (R + H*P*H');
                fS = self.fP[j] + fR

                # sP^T = sP, sR^t = sR -> sS^-1*sP = sP*sS^-1
                sK = np.linalg.solve(sS, self.sP[j]) # K = P*H'/IS;
                fK = np.linalg.solve(fS, self.fP[j])

                pot_sm[k, j] = self.sm[j] + np.dot(sK, sy) # X = X + K * (y-IM);
                pot_fm[k, j] = self.fm[j] + np.dot(fK, fy)
                pot_sP[k, j] = np.dot((np.identity(spatial_dim) - sK), self.sP[j])
                pot_fP[k, j] = np.dot((np.identity(feature_dim) - fK), self.fP[j])

                # TODO: wait 1s, didn't they write that this should be pot_sm[j]?
                spatial_likelihoods[k, j] = gauss_pdf(spatial_measurements[k], self.sm[j], sS)
                feature_likelihoods[k, j] = gauss_pdf(feature_measurements[k], self.fm[j], fS)

            likelihoods[k, :nbr_targets] = spatial_likelihoods[k, :]*feature_likelihoods[k, :]
            likelihoods[k, nbr_targets] = pdclutter
            #likelihoods = 1./np.sum(likelihoods)*likelihoods

            # probability of measurement given association, may arise from clutter anyways?
            pc[k, :nbr_targets] = (1.-pclutter)*likelihoods[k, :nbr_targets]
            pc[k, nbr_targets] = pclutter*likelihoods[k, nbr_targets]
            pc[k, nbr_targets+1] = pjump
            pc[k] = 1./np.sum(pc[k])*pc[k]

            #for picked in self.c:
            #    pc[k, picked] = 0.
            #pc = 1./np.sum(pc)*pc

        with_cartesian = False

        if with_cartesian:
            # now we need to do a cartesian combination of the likelihoods and set indices
            # where assignment overlap to 0
            joint_pcs = np.zeros((1, 1))
            for k in range(0, nbr_observations):
                joint_pcs = np.kron(joint_pcs, pc[k])
            joint_pcs = joint_pcs.flatten()
            # we need to go through each pair i, i and set it to zero...
            for j in range(0, nbr_targets):
                inds = cartesian_inds.overlapping_indices_for_state(j, nbr_targets, nbr_observations)
                print joint_pcs.shape
                print inds.shape
                joint_pcs[inds] = 0.

            # rejection sampling would be quite a lot easier and probably faster
            # hmm, that might also lead to a situation where we can avoid cartesian combinations alltogether
            print (nbr_targets+2)**nbr_observations
            cartesian_ind = np.random.choice((nbr_targets+2)**nbr_observations, p=pc) #categ_rnd()
            weights_update = 1.
            states = cartesian_inds.cartesian_to_states(cartesian_ind, nbr_targets, nbr_observations)
        else:
            states = np.zeros((nbr_observations,), dtype=int)
            while True:
                for k in range(0, nbr_observations):
                    states[k] = np.random.choice(nbr_targets+2, p=pc[k])
                unique, counts = np.unique(states, return_counts=True)
                unique_list = unique.tolist()
                nbr_sampled_noise = 0
                if nbr_targets in unique_list:
                    nbr_sampled_noise = counts[unique_list.index(nbr_targets)] - 1
                print "Nbr sampled noise, targets: ", nbr_sampled_noise, nbr_targets
                if len(unique) + nbr_sampled_noise == nbr_observations:
                    break
                else:
                    print "Found several similar states: ", states

        weights_update = 1.

        for k, i in enumerate(states):

            # so what happens here if i ==
            if i == nbr_targets+1:
                weights_update *= pjump
            else:
                weights_update *= likelihoods[k, i]/pc[k, i]

            if i == nbr_targets:
                pass
                #self.c = nbr_targets # measurement is noise
                #we don't really care if it was associated with noise

            elif i == nbr_targets+1:
                feature_likelihoods_k = feature_likelihoods[k]/np.sum(feature_likelihoods[k])
                i = np.random.choice(nbr_targets, p=feature_likelihoods_k)
                self.sm[i] = spatial_measurements[k]
                self.sP[i] = 1.0*np.eye(spatial_dim)
            else:
                self.sm[i] = pot_sm[k, i]
                self.fm[i] = pot_fm[k, i]
                self.sP[i] = pot_sP[k, i]
                self.fP[i] = pot_fP[k, i]
                self.c.append(i)
                self.associations[observation_id] = i

        return weights_update
