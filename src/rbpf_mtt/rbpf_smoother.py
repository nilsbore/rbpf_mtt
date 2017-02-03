from rbpf_mtt.rbpf_filter import RBPFMTTFilter, RBPFMParticle, gauss_pdf
import numpy as np
import math

# because the filter doesn't save its state, we will need to delegate the
# functions to the filter as a member
class RBPFMTTSmoother(object):

    def __init__(self, nbr_targets, nbr_particles, feature_dim, nbr_timesteps, nbr_backward_sims):

        self.nbr_particles = nbr_particles
        self.nbr_timesteps = nbr_timesteps
        self.nbr_backward_sims = nbr_backward_sims

        self.filter = RBPFMTTFilter(nbr_targets, nbr_particles, feature_dim)

        self.particles = [[] for i in range(0, nbr_timesteps)]
        self.timestep_weights = 1./nbr_particles*np.ones((nbr_timesteps, nbr_particles))
        self.spatial_measurements = np.zeros((nbr_timesteps, 2))
        self.feature_measurements = np.zeros((nbr_timesteps, feature_dim))

    def resample(self):

        self.filter.resample()

    def single_update(self, spatial_measurement, feature_measurement, time, observation_id):

        self.filter.single_update(spatial_measurement, feature_measurement, time, observation_id)

    def predict(self):

        self.filter.predict()

    def initialize_target(self, target_id, spatial_measurement, feature_measurement):

        self.filter.initialize_target(target_id, spatial_measurement, feature_measurement)

    # now, we should already have all the information, let's just roll this
    # backwards and see what happens!
    def smooth(self):

        for s in range(0, self.nbr_backward_sims):

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

                smk =

                # Smoother weights update
                # w^(i)_k|k+1 ∝ w^(i)_k p(u^~_k+1 | u^(i)_k)  | det A(u^(i)_k)|^-1
                # * N(m^(i)_k| m^-b,(i)_k, P^(i)_k + P^-b,(i)_k)

                # Sample u^~_k = u^(i)_k with probability w^(i)_k|k+1
