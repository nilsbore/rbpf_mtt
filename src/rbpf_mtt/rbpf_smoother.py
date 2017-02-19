from rbpf_mtt.rbpf_filter import RBPFMTTFilter
from rbpf_mtt.rbpf_particle import RBPFMParticle, gauss_pdf
import numpy as np
import math
import copy
import sys

# because the filter doesn't save its state, we will need to delegate the
# functions to the filter as a member
class RBPFMTTSmoother(object):

    def __init__(self, nbr_targets, nbr_particles, feature_dim, nbr_backward_sims):

        self.nbr_targets = nbr_targets
        self.nbr_particles = nbr_particles
        self.nbr_backward_sims = nbr_backward_sims
        self.feature_dim = feature_dim

        max_iterations = 1000

        self.filter = RBPFMTTFilter(nbr_targets, nbr_particles, feature_dim)

        self.timestep_particles = [[] for i in range(0, max_iterations)]
        self.timestep_weights = 1./nbr_particles*np.ones((max_iterations, nbr_particles))
        self.spatial_measurements = np.zeros((max_iterations, 2))
        self.feature_measurements = np.zeros((max_iterations, feature_dim))
        self.timesteps = np.zeros((max_iterations,), dtype=int)
        self.cindex = np.zeros((max_iterations,), dtype=int)

        self.nbr_timesteps = 0

        self.debug = False

    def resample(self):

        self.filter.resample()

    def single_update(self, spatial_measurement, feature_measurement, time, observation_id):

        self.spatial_measurements[self.nbr_timesteps] = spatial_measurement
        self.feature_measurements[self.nbr_timesteps] = feature_measurement
        self.timesteps[self.nbr_timesteps] = time
        self.filter.single_update(spatial_measurement, feature_measurement, time, observation_id)
        #self.timestep_particles[self.nbr_timesteps] = copy.deepcopy(self.filter.particles)
        self.timestep_particles[self.nbr_timesteps] = copy.deepcopy(self.filter.particles)
        self.timestep_weights[self.nbr_timesteps] = np.array(self.filter.weights)
        self.nbr_timesteps += 1
        self.cindex[self.nbr_timesteps] = 0

    def joint_update(self, spatial_measurements, feature_measurements, time, observation_id):

        self.filter.joint_update(spatial_measurements, feature_measurements, time, observation_id)

        # seems a bit unnecessary to save two copies of every particle, also we don't actually
        # update the kalman filters until after all of the observations are integrated ...
        # but, it might not matter since we only integrate into the filter that actually had a measurement
        # so, actually it should be find. This has the added advantage that the smoothing will
        # work the same way irrespective of if we do the single update or the joint update
        for k in range(0, spatial_measurements.shape[0]):
            self.spatial_measurements[self.nbr_timesteps] = spatial_measurements[k]
            self.feature_measurements[self.nbr_timesteps] = feature_measurements[k]
            self.timesteps[self.nbr_timesteps] = time
            self.timestep_particles[self.nbr_timesteps] = copy.deepcopy(self.filter.particles)
            self.timestep_weights[self.nbr_timesteps] = np.array(self.filter.weights)
            self.cindex[self.nbr_timesteps] = k
            for i in range(0, self.nbr_particles):
                if len(self.filter.particles[i].c) <= k:
                    print "Error!", k, self.filter.particles[i].c
                    sys.exit()
            self.nbr_timesteps += 1

    def predict(self):

        self.filter.predict()

    def initialize_target(self, target_id, spatial_measurement, feature_measurement, time):

        self.spatial_measurements[self.nbr_timesteps] = spatial_measurement
        self.feature_measurements[self.nbr_timesteps] = feature_measurement
        self.timesteps[self.nbr_timesteps] = time
        self.filter.initialize_target(target_id, spatial_measurement, feature_measurement)
        #self.timestep_particles[self.nbr_timesteps] = copy.deepcopy(self.filter.particles)
        self.timestep_particles[self.nbr_timesteps] = copy.deepcopy(self.filter.particles)
        self.timestep_weights[self.nbr_timesteps] = np.array(self.filter.weights)
        self.cindex[self.nbr_timesteps] = target_id
        self.nbr_timesteps += 1

    def backwards_update(self, k, smbk1, sPbk1, fmbk1, fPbk1, cb):

        spatial_dim = 2
        feature_dim = self.feature_dim
        nbr_targets = self.nbr_targets

        pclutter = 0.
        pjump = 0.01

        spatial_var = self.filter.spatial_std*self.filter.spatial_std
        feature_var = self.filter.feature_std*self.filter.feature_std

        sR = spatial_var*np.identity(spatial_dim) # measurement noise
        fR = feature_var*np.identity(feature_dim) # measurement noise

        spatial_process_noise = 0. # no process noise!
        feature_process_noise = 0. # no process noise!

        sQ = spatial_process_noise*np.identity(spatial_dim) # process noise
        fQ = feature_process_noise*np.identity(feature_dim) # process noise

        # NOTE: DONE
        wm = np.zeros((self.nbr_particles,))
        debugll = np.zeros((self.nbr_particles,))
        debugpc = np.zeros((self.nbr_particles,))
        # we need to save smbk, fmbk, sPbk, fPbk to compute next iteration and smoothed estimates
        # do we need these for every target? I guess so...
        smb = np.zeros((self.nbr_particles, spatial_dim))
        sPb = np.zeros((self.nbr_particles, spatial_dim, spatial_dim))
        fmb = np.zeros((self.nbr_particles, feature_dim))
        fPb = np.zeros((self.nbr_particles, feature_dim, feature_dim))
        tb = np.zeros((self.nbr_particles,), dtype=int)

        # NOTE: DONE
        for l in range(0, self.nbr_particles):

            # print self.timesteps[k], self.timesteps[k+1]
            # print self.nbr_timesteps, k, len(self.cindex), l
            # print self.cindex[k]
            # print len(self.timestep_particles[k])
            # print self.timestep_particles[k][l].c
            if len(self.timestep_particles[k][l].c) <= self.cindex[k] and self.debug:
                 print "Count: ", np.sum(self.timesteps == self.timesteps[k])
                 print self.timesteps[k]
                 print self.timesteps
                 print self.cindex[k]
                 print len(self.timestep_particles[k][l].c)
            cj = self.timestep_particles[k][l].c[self.cindex[k]]
            tj = self.timesteps[k]
            j = cj % (nbr_targets+1)

            if cj == nbr_targets: # noise!
                # here we should just propagate the previous backward estimates
                pc = pclutter
                likelihoodb = 1.
                debugll[l] = 1.
                tb[l] = -2
            # NOTE: this is not enough as there may be overlap in previous associations as well
            #elif ti == tj and cj % (nbr_targets+1) == ci % (nbr_targets+1): # can't associate same target twice!
            elif j in cb:
                # in this case, this won't be sampled and we don't need to worry about the estimates
                #print j, " in ", cb
                pc = 0.
                likelihoodb = 1.
                debugll[l] = 2.
                tb[l] = -1
            else:
                #print j, " not in ", cb
                # This is the equations for the backwards filter, with simplifications
                # m^-b_k = A(u_k) m_k+1^b = m_k+1^b
                # P_k^-b = A^-1(u_k) (Q(u_k) + P_k+1^b) A^-T(u_k) = P_k+1^b (+ Q(u_k))

                # \mu^b_k = H(u_k) m^-b_k = H(u_k) m^b_k+1 = m^b_k+1
                # S_k^b = H(u_k) P_k^-b H^T(u_k) + R(u_k) = H(u_k) P_k+1^b H^T(u_k) + R(u_k) = P_k+1^b + R(u_k)
                # K_k_b = P_k^-b H^T(u_k) (S_k^b)^-1 = P_k+1^b H^T(u_k) (S_k^b)^-1 = P_k+1^b (S_k^b)^-1
                # m^b_k = m^-b_k + K_k^b [y_k - \mu_k^b] = m^b_k+1 + K_k^b [y_k - \mu_k^b]
                # P_k^b = P_k^-b - K_k^b S_k^b (K_k^b)^T = P_k+1^b - K_k^b S_k^b (K_k^b)^T
                # Z_k = Z_k+1 |det A(u_k )|^-1 N(y_k |\mu^b_k , S_k^b) = Z_k+1 N(y_k |\mu^b_k , S_k^b)

                # w_k|k+1 = w_k p(u~_k+1 | u_k) N(m_k | m_k+1^b, P_k + P_k+1^b)

                #fPbkm1 = fPbk1[j] + fQ
                fmubk = fmbk1[j]
                fSbk = fPbk1[j] + fR
                fKbk = np.linalg.solve(fSbk, fPbk1[j])
                #fKbk = np.linalg.solve(fSbk, fPbk1[j])
                fmbk = fmbk1[j] + np.dot(fKbk, self.feature_measurements[k] - fmubk)
                fPbk = fPbk1[j] - np.dot(fKbk, np.dot(fSbk, fKbk.transpose()))
                #fZk = fZ[k+1] * gauss_pdf(self.feature_measurements[k], fmubk, fSbk)

                fmb[l] = fmbk
                fPb[l] = fPbk

                # These are the smoothed estimates from combining the backward and forward filter
                fmk = np.array(self.timestep_particles[k][l].fm[j])
                fPk = np.array(self.timestep_particles[k][l].fP[j])

                #print fmk, fmbk1[j], fPk + fPbk1[j]
                #print np.linalg.det(fPk + fPbk1[j])
                likelihoodb = gauss_pdf(fmk, fmbk1[j], fPk + fPbk1[j])
                debugll[l] = likelihoodb

                tb[l] = j

                if cj < nbr_targets: # normal association!

                    smubk = smbk1[j]
                    sSbk = sPbk1[j] + sR
                    sKbk = np.linalg.solve(sSbk, sPbk1[j])
                    smbk = smbk1[j] + np.dot(sKbk, self.spatial_measurements[k] - smubk)
                    sPbk = sPbk1[j] - np.dot(sKbk, np.dot(sSbk, sKbk.transpose()))
                    #sZk = sZ[k+1] * gauss_pdf(self.spatial_measurements[k], smubk, sSbk)

                    smb[l] = smbk
                    sPb[l] = sPbk

                    smk = np.array(self.timestep_particles[k][l].sm[j])
                    sPk = np.array(self.timestep_particles[k][l].sP[j])

                    likelihoodb *= gauss_pdf(smk, smbk1[j], sPk + sPbk1[j])
                    debugpc[l] = gauss_pdf(smk, smbk1[j], sPk + sPbk1[j])

                    pc = 1.#(1.-pjump)/float(nbr_targets)

                else: # jump association!
                    # how do we handle the backward estimates of m, P here?
                    # it seems like we should simply set the estimates to the measurement + noise
                    smb[l] = self.spatial_measurements[k]
                    sPb[l] = sR
                    pc = pjump

            # Compute w_k|k+1 = w_k p(u~_k+1 | u_k) N(m_k | m_k+1^b, P_k + P_k+1^b)
            wm[l] = self.timestep_weights[k, l] * pc * likelihoodb

        wmsum = np.sum(wm)

        # Sample u^~_k = u^(i)_k with probability w^(i)_k|k+1
        #print "Sum: ", wmsum

        if wmsum == 0:
            if self.debug:
                for j in range(0, nbr_targets):
                    print np.linalg.det(fPbk1[j] + self.timestep_particles[k][0].fP[j])
                    print fmbk1[j]
                    print fPbk1[j]
                    print self.timestep_particles[k][0].fm[j]
                    print self.timestep_particles[k][0].fP[j]
                    #print smbk1[j]
                    #print sPbk1[j]
                #print wm
                print debugll
                print debugpc
            return -1, -1, smbk1, sPbk1, fmbk1, fPbk1, cb

        wm = 1./wmsum*wm

        i = np.random.choice(self.nbr_particles, p=wm)
        j = tb[i]
        cb.append(j)

        # Store the backwards estimates for the previous timestep
        smbk1[j] = smb[i]
        sPbk1[j] = sPb[i]
        fmbk1[j] = fmb[i]
        fPbk1[j] = fPb[i]

        #print i, j, k, self.nbr_timesteps, wm[i]
        #print sPbk1[j], sPk
        if j < 0:
            print wm

        return i, j, smbk1, sPbk1, fmbk1, fPbk1, cb

    def compute_smoothed_estimates(self, k, i, j, smbk1, sPbk1, fmbk1, fPbk1):

        # All of the following code is for computing the smoothed estimates
        smk = self.timestep_particles[k][i].sm[j]
        sPk = self.timestep_particles[k][i].sP[j]

        # These are the smoothed estimates from combining the backward and forward filter
        sPsk = np.linalg.inv(np.linalg.inv(sPbk1[j]) + np.linalg.inv(sPk))
        smsk = np.dot(sPsk, np.linalg.solve(sPk, smk) + np.linalg.solve(sPbk1[j], smbk1[j]))

        fmk = self.timestep_particles[k][i].fm[j]
        fPk = self.timestep_particles[k][i].fP[j]

        # These are the smoothed estimates from combining the backward and forward filter
        fPsk = np.linalg.inv(np.linalg.inv(fPbk1[j]) + np.linalg.inv(fPk))
        fmsk = np.dot(fPsk, np.linalg.solve(fPk, fmk) + np.linalg.solve(fPbk1[j], fmbk1[j]))

        # store the smoothed estimates for this timestep
        return smsk, sPsk, fmsk, fPsk


    # now, we should already have all the information, let's just roll this
    # backwards and see what happens!
    def smooth(self):

        spatial_dim = 2
        feature_dim = self.feature_dim
        nbr_targets = self.nbr_targets

        # smoothed estimates!
        self.sms = np.zeros((self.nbr_backward_sims, self.nbr_timesteps, nbr_targets, spatial_dim))
        self.sPs = np.zeros((self.nbr_backward_sims, self.nbr_timesteps, nbr_targets, spatial_dim, spatial_dim))
        self.fms = np.zeros((self.nbr_backward_sims, self.nbr_timesteps, nbr_targets, feature_dim))
        self.fPs = np.zeros((self.nbr_backward_sims, self.nbr_timesteps, nbr_targets, feature_dim, feature_dim))
        self.cs = np.zeros((self.nbr_backward_sims, self.nbr_timesteps), dtype=int)

        # NOTE: DONE
        for s in range(0, self.nbr_backward_sims):

            print "Running backward simulation: ", s

            # sample the latent state with probability of the last weights
            # weights are always normalized in the update equation
            # NOTE: here we just sample 1 particle
            # NOTE: DONE
            i = np.random.choice(self.nbr_particles, p=self.filter.weights)
            #ci = self.timestep_particles[self.nbr_timesteps-1][i].c[self.cindex[self.nbr_timesteps-1]]
            #ti = self.timesteps[self.nbr_timesteps-1]

            smbk1 = np.array(self.timestep_particles[self.nbr_timesteps-1][i].sm) #np.zeros((self.nbr_targets, spatial_dim))
            sPbk1 = np.array(self.timestep_particles[self.nbr_timesteps-1][i].sP) #np.zeros((self.nbr_targets, spatial_dim, spatial_dim))
            fmbk1 = np.array(self.timestep_particles[self.nbr_timesteps-1][i].fm) #np.zeros((self.nbr_targets, feature_dim))
            fPbk1 = np.array(self.timestep_particles[self.nbr_timesteps-1][i].fP) #np.zeros((self.nbr_targets, feature_dim, feature_dim))
            for j in range(0, nbr_targets):
                self.sms[s, :, j] = np.array(smbk1[j])
                self.sPs[s, :, j] = np.array(sPbk1[j])
                self.fms[s, :, j] = np.array(fmbk1[j])
                self.fPs[s, :, j] = np.array(fPbk1[j])

            cache_i, cache_smbk1, cache_sPbk1, cache_fmbk1, cache_fPbk1 = \
                i, np.array(smbk1), np.array(sPbk1), np.array(fmbk1), np.array(fPbk1)
            current_timestep = self.timesteps[self.nbr_timesteps-1]
            start_timestep = self.nbr_timesteps-1
            cb = [] # self.timestep_particles[self.nbr_timesteps-1][i].c[self.cindex[self.nbr_timesteps-1]] % (nbr_targets+1) ] # this assumes not noise!

            k = self.nbr_timesteps-1
            while k >= 0:

                print "Time: ", k
                print "Meas Time: ", self.timesteps[k]
                print "C index: ", self.cindex[k]
                print "cb: ", cb

                if self.timesteps[k] != current_timestep:
                    print "New timestep: ", self.timesteps[k]
                    # update the targets which have not been updated, i.e. noise assocs
                    # actually, we don't need to, they are already in the cached previous ones
                    cb = []
                    cache_i, cache_smbk1, cache_sPbk1, cache_fmbk1, cache_fPbk1 = \
                        i, np.array(smbk1), np.array(sPbk1), np.array(fmbk1), np.array(fPbk1)
                    current_timestep = self.timesteps[k]
                    start_timestep = k

                # this might actually happen if we have not sampled enough associations,
                # the solution is to repeat sampling for this timestep...
                i, j, smbk1, sPbk1, fmbk1, fPbk1, cb = self.backwards_update(k, smbk1, sPbk1, fmbk1, fPbk1, cb)
                if j == -1: # assume this can't happen for now, re-run this timestep
                    print "Re-running iteration: ", k
                    k = start_timestep
                    i, smbk1, sPbk1, fmbk1, fPbk1 = \
                        np.array(cache_i), np.array(cache_smbk1), np.array(cache_sPbk1), np.array(cache_fmbk1), np.array(cache_fPbk1)
                    if start_timestep == self.nbr_timesteps-1:
                        #print "First assoc: ", self.timestep_particles[self.nbr_timesteps-1][i].c[self.cindex[self.nbr_timesteps-1]]
                        cb = [] #[ self.timestep_particles[self.nbr_timesteps-1][i].c[self.cindex[self.nbr_timesteps-1]] % (nbr_targets+1) ] # this assumes not noise!
                    else:
                        cb = []
                    continue

                smsk, sPsk, fmsk, fPsk = self.compute_smoothed_estimates(k, i, j, smbk1, sPbk1, fmbk1, fPbk1)
                self.sms[s, :k+1, j] = np.array(smsk)
                self.sPs[s, :k+1, j] = np.array(sPsk)
                self.fms[s, :k+1, j] = np.array(fmsk)
                self.fPs[s, :k+1, j] = np.array(fPsk)
                self.cs[s, k] = self.timestep_particles[k][i].c[self.cindex[k]]

                k -= 1
