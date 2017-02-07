from rbpf_mtt.rbpf_filter import RBPFMTTFilter, RBPFMParticle, gauss_pdf
import numpy as np
import math
import copy

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

        spatial_dim = 2
        feature_dim = self.feature_dim

        spatial_measurement_noise = 0.4
        feature_measurement_noise = 0.2

        sR = spatial_measurement_noise*np.identity(spatial_dim) # measurement noise
        fR = feature_measurement_noise*np.identity(feature_dim) # measurement noise

        spatial_process_noise = 1.5
        feature_process_noise = 0.1

        sQ = np.identity(spatial_dim) # process noise
        fQ = np.identity(feature_dim) # process noise

        for s in range(0, self.nbr_backward_sims):

            print "Running backward simulation: ", s

            # sample the latent state with probability of the last weights
            # weights are always normalized in the update equation
            i = np.random.choice(self.nbr_particles, p=self.filter.weights)
            for k in range(self.nbr_timesteps-1, -1, -1):
                # Computing the Kalman filter recursions
                # Note: A - process update = I
                # m^-1_k = A(u_k-1) m_k-1
                # Note: A = I
                # Note: Q - process noise
                # P^-_k = A(u_k-1) P_k-1 A^T(u_k-1) + Q(u_k-1)
                # Note: H - measurement process = I
                # Note: R - measurement noise
                # S_k = H(uk) P-_k H^T(u_k) + R(u_k)
                # Kk = P^-_k H^T(u_k) S^-1_k
                # Note - the measurement comes in here
                # m_k = m^-_k + K_k [y_k - H(u_k) m^-_k]
                # P_k = P^-_k - K_k S_k K^T_k

                smM = np.zeros((self.nbr_targets, spatial_dim)) # the Kalman filter means
                fmM = np.zeros((self.nbr_targets, feature_dim))
                sPM = np.zeros((self.nbr_targets, spatial_dim, spatial_dim)) # the Kalman filter covariances
                fPM = np.zeros((self.nbr_targets, feature_dim, feature_dim))
                wM = np.zeros((self.nbr_particles,))

                for l in range(0, self.nbr_particles):

                    # for j in range(0, nbr_targets):
                    #     smm = self.timestep_particles[k][i].sm[j]
                    #     sPm = self.timestep_particles[k][i].sP[j] + sQ
                    #     sSm = sPm + sR
                    #     sKm = np.solve(sSm, sPm)
                    #     # the question is, what is meas here?
                    #     # hmm, I guess our measurement model might actually depend on c..., i.e. 0 for some
                    #     # or rather, how can we motivate this being the meas indicated by c?
                    #     # right, we only update the part indicated by c, so the measurement model
                    #     # actually transforms down the big state vector quite a bit
                    #     sm = smm + np.dot(smK, meas - smm)
                    #     sP = sPm - np.dot(sKm, np.dot(sSm, sKm.transpose()))

                    # NOTE: this is exactly the kalman update equations

                    #print k, l
                    #print len(self.timestep_particles), len(self.timestep_particles[k]), len(c)
                    if len(self.timestep_particles[k][l].c) == 0:
                        pclutter = 0.01
                        wM[l] = self.timestep_weights[k, l] * pclutter
                        continue

                    j = self.timestep_particles[k][l].c[-1] # this should be the target associated with this measurement
                    # TODO: of course, we don't know this since it might also be that it was associated with noise
                    smm = self.timestep_particles[k][l].sm[j]
                    # this is the predict update
                    sPm = self.timestep_particles[k][l].sP[j] + sQ
                    sSm = sPm + sR
                    # this is basically the same as the update
                    sKm = np.linalg.solve(sSm, sPm)
                    # the question is, what is meas here?
                    # hmm, I guess our measurement model might actually depend on c..., i.e. 0 for some
                    # or rather, how can we motivate this being the meas indicated by c?
                    # right, we only update the part indicated by c, so the measurement model
                    # actually transforms down the big state vector quite a bit
                    sm = smm + np.dot(sKm, self.spatial_measurements[k] - smm)
                    sP = sPm - np.dot(sKm, np.dot(sSm, sKm.transpose()))

                    fmm = self.timestep_particles[k][l].fm[j]
                    fPm = self.timestep_particles[k][l].fP[j] + fQ
                    fSm = fPm + fR
                    fKm = np.linalg.solve(fSm, fPm)
                    # the question is, what is meas here?
                    # hmm, I guess our measurement model might actually depend on c..., i.e. 0 for some
                    # or rather, how can we motivate this being the meas indicated by c?
                    # right, we only update the part indicated by c, so the measurement model
                    # actually transforms down the big state vector quite a bit
                    fm = fmm + np.dot(fKm, self.feature_measurements[k] - fmm)
                    fP = fPm - np.dot(fKm, np.dot(fSm, fKm.transpose()))

                    # This is interesting, now we need to compute p(u^~_k+1 | u^(i)_k)
                    # What is u^~_k+1 ???
                    # Oh, right, it's the sampled latent variable of the previous iteration
                    # so, in other words, we need to compute this probability for every particle
                    # do we also need to compute the kalman covariances for every particle? probably...

                    pc = 0.
                    # so, probability of i given l
                    if i in self.timestep_particles[k][l].c:
                        #0! unless, TODO: the timestep is different for i...
                        pc = 0.
                    else:
                        # to compute this, we need to look at the measurement likelihoods...
                        # it would probably be good to save them for this reason...
                        # the reason for this is that we don't know the normalization...
                        # so, in reality, we could actually limit ourselves to saving normalization?
                        # I think no, because we need the Kalman updates to compute the likelihoods?
                        # p(i | l)
                        # this is actually only part of it, we need the clutter and jump stuff as well
                        pc = gauss_pdf(self.spatial_measurements[k], self.timestep_particles[k][l].sm[j], sSm) * \
                             gauss_pdf(self.feature_measurements[k], self.timestep_particles[k][l].fm[j], fSm)

                    # Smoother weights update
                    # w^(i)_k|k+1 \sim w^(i)_k p(u^~_k+1 | u^(i)_k)  |det A(u^(i)_k)|^-1
                    # * N(m^(i)_k| m^-b,(i)_k, P^(i)_k + P^-b,(i)_k)
                    # The question arises again, which measurement should we use?

                    # TODO: check if it's actually these sm, fm, sP, fP that we should use
                    wM[l] = self.timestep_weights[k, l] * pc * \
                            gauss_pdf(self.timestep_particles[k][l].sm[j], sm,
                                      self.timestep_particles[k][l].sP[j] + sP) * \
                            gauss_pdf(self.timestep_particles[k][l].fm[j], fm,
                                      self.timestep_particles[k][l].fP[j] + fP)

                wM = 1./np.sum(wM)*wM
                # Sample u^~_k = u^(i)_k with probability w^(i)_k|k+1
                i = np.random.choice(self.nbr_particles, p=wM)
