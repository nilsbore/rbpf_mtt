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
            self.nbr_timesteps += 1
            self.cindex[self.nbr_timesteps] = k

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
        self.nbr_timesteps += 1

    # now, we should already have all the information, let's just roll this
    # backwards and see what happens!
    def smooth(self):

        pclutter = 0.0
        pjump = 0.001

        spatial_dim = 2
        feature_dim = self.feature_dim
        nbr_targets = self.nbr_targets

        spatial_var = self.filter.spatial_std*self.filter.spatial_std
        feature_var = self.filter.feature_std*self.filter.feature_std

        sR = spatial_var*np.identity(spatial_dim) # measurement noise
        fR = feature_var*np.identity(feature_dim) # measurement noise

        spatial_process_noise = 0. # no process noise!
        feature_process_noise = 0. # no process noise!

        sQ = np.identity(spatial_dim) # process noise
        fQ = np.identity(feature_dim) # process noise

        # NOTE: DONE
        for s in range(0, self.nbr_backward_sims):

            print "Running backward simulation: ", s

            # sample the latent state with probability of the last weights
            # weights are always normalized in the update equation
            # NOTE: here we just sample 1 particle
            # NOTE: DONE
            i = np.random.choice(self.nbr_particles, p=self.filter.weights)
            ci = self.timestep_particles[self.nbr_timesteps-1][i].c[self.cindex[self.nbr_timesteps-1]]
            ti = self.timesteps[self.nbr_timesteps-1]
            for k in range(self.nbr_timesteps-1, -1, -1):

                # NOTE: DONE
                wm = np.zeros((self.nbr_particles,))

                # NOTE: DONE
                for l in range(0, self.nbr_particles):

                    # NOTE: this is exactly the kalman update equations

                    cj = self.timestep_particles[k][l].c[self.cindex[k]]
                    tj = self.timesteps[k]

                    if cj == nbr_targets: # noise!
                        pc = pclutter
                        mlikelihood = 1.
                    elif ti == tj and cj % (nbr_targets+1) == ci % (nbr_targets+1):
                        pc = 0
                        mlikelihood = 1.
                    else:
                        j = cj % (nbr_targets+1)

                        # This is the equations for the backwards filter, with simplifications
                        # m^-b_k = A(u_k) m_k+1^b = m_k+1^b
                        # P_k^-b = A^-1(u_k) (Q(u_k) + P_k+1^b) A^-T(u_k) = P_k+1^b

                        # \mu^b_k = H(u_k) m^-b_k = H(u_k) m^b_k+1 = m^b_k+1
                        # S_k^b = H(u_k) P_k^-b H^T(u_k) + R(u_k) = H(u_k) P_k+1^b H^T(u_k) + R(u_k) = P_k+1^b + R(u_k)
                        # K_k_b = P_k^-b H^T(u_k) (S_k^b)^-1 = P_k+1^b H^T(u_k) (S_k^b)^-1 = P_k+1^b (S_k^b)^-1
                        # m^b_k = m^-b_k + K_k^b [y_k - \mu_k^b] = m^b_k+1 + K_k^b [y_k - \mu_k^b]
                        # P_k^b = P_k^-b - K_k^b S_k^b (K_k^b)^T = P_k+1^b - K_k^b S_k^b (K_k^b)^T
                        # Z_k = Z_k+1 |det A(u_k )|^-1 N(y_k |\mu^b_k , S_k^b) = Z_k+1 N(y_k |\mu^b_k , S_k^b)

                        # w_k|k+1 = w_k p(u~_k+1 | u_k) N(m_k | m_k+1^b, P_k + P_k+1^b)

                        smubk = smb[k+1]
                        sSbk = sPb[k+1] + sR
                        sKbk = np.linalg.solve(sPb[k+1], sPb[k+1])
                        smbk = smb[k+1] + np.dot(sKbk, self.spatial_measurements[k] - smubk)
                        sPbk = sPb[k+1] - np.dot(sKbk, np.dot(sSbk, sKbk.transpose()))
                        sZk = sZ[k+1] * gauss_pdf(self.spatial_measurements[k], smubk, sSbk)

                        smk = self.timestep_particles[k][l].sm[j]
                        sPk = self.timestep_particles[k][l].sP[j]

                        likelihoodb = gauss_pdf(smk, smb[k+1], sPk + sPb[k+1])

                        # These are the smoothed estimates from combining the backward and forward filter
                        sPsk = (sPbk.inverse() + sPk.inverse()).inverse()
                        smsk = np.dot(sPsk, np.linalg.solve(sPk, smk) + np.linalg.solve(sPbk, smbk))

                        if cj < nbr_targets: # normal association!
                            fmubk = fmb[k+1]
                            fSbk = fPb[k+1] + fR
                            fKbk = np.linalg.solve(fPb[k+1], fPb[k+1])
                            fmbk = fmb[k+1] + np.dot(fKbk, self.feature_measurements[k] - fmubk)
                            fPbk = fPb[k+1] - np.dot(fKbk, np.dot(fSbk, fKbk.transpose()))
                            fZk = fZ[k+1] * gauss_pdf(self.feature_measurements[k], fmubk, fSbk)

                            fmk = self.timestep_particles[k][l].fm[j]
                            fPk = self.timestep_particles[k][l].fP[j]

                            likelihoodb *= gauss_pdf(fmk, fmb[k+1], fPk + fPb[k+1])

                            fPsk = (fPbk.inverse() + fPk.inverse()).inverse()
                            fmsk = np.dot(fPsk, np.linalg.solve(fPk, fmk) + np.linalg.solve(fPbk, fmbk))

                            pc = (1.-pjump)/float(nbr_targets)
                        else: # jump association!
                            pc = pjump


                    # TODO: check if it's actually these sm, fm, sP, fP that we should use
                    wm[l] = self.timestep_weights[k, l] * pc * likelihoodb


                wm = 1./np.sum(wm)*wm
                # Sample u^~_k = u^(i)_k with probability w^(i)_k|k+1
                i = np.random.choice(self.nbr_particles, p=wm)
