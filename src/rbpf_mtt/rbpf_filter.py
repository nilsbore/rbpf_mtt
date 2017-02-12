import numpy as np
import math
import copy
from cartesian import cartesian_inds
from rbpf_mtt.rbpf_particle import RBPFMParticle
import random
import matplotlib.pyplot as plt

class RBPFMTTFilter(object):

    def __init__(self, nbr_targets, nbr_particles, feature_dim, spatial_std=0.63, feature_std=0.45):

        self.dim = 2 # I would like for this to be global instead
        self.feature_dim = feature_dim
        self.nbr_targets = nbr_targets
        self.nbr_particles = nbr_particles
        self.spatial_std = spatial_std
        self.feature_std = feature_std

        self.particles = [RBPFMParticle(self.dim, feature_dim, nbr_targets, spatial_std, feature_std) for i in range(0, nbr_particles)]
        self.weights = 1./nbr_particles*np.ones((nbr_particles))

        self.last_time = -1

        self.resampled = False
        self.time_since_resampling = 0

    def effective_sample_size(self):

        return 1./np.sum(np.square(self.weights)) # should this also be sqrt? no!

    def multinomial_resample(self): # maybe this should take measurements?

        old_particles = copy.deepcopy(self.particles)

        samples = np.random.choice(self.nbr_particles, self.nbr_particles, p=self.weights, replace=True)

        print "Weights: ", self.weights
        print "Samples: ", samples

        # now resample based on the weights
        for i in range(0, self.nbr_particles):
            self.weights[i] = 1./float(self.nbr_particles)
            self.particles[i] = copy.deepcopy(old_particles[samples[i]])

        self.resampled = True
        self.time_since_resampling = 0

    def systematic_resample(self): # maybe this should take measurements?

        old_particles = copy.deepcopy(self.particles)

        #samples = np.random.choice(self.nbr_particles, self.nbr_particles, p=self.weights, replace=True)



        print "Weights: ", self.weights
        #print "Samples: ", samples

        u = random.random()
        U = np.arange(self.nbr_particles, dtype=float)/float(self.nbr_particles) + u
        wc = np.cumsum(self.weights)

        samples = np.zeros((self.nbr_particles,), dtype=int)
        k = 0
        for i in range(0, self.nbr_particles):
            while k < self.nbr_particles-1 and wc[k] < U[i]:
                k += 1
            samples[i] = k

        # now resample based on the weights
        for i in range(0, self.nbr_particles):
            self.weights[i] = 1./float(self.nbr_particles)
            self.particles[i] = old_particles[samples[i]]

        self.resampled = True
        self.time_since_resampling = 0

    # should this take a time interval also?
    def predict(self):

        for p in self.particles:
            p.predict()


    def single_update(self, spatial_measurement, feature_measurement, time, observation_id):

        if time != self.last_time:
            pass#self.predict()

        #if self.last_time != time and self.last_time != -1:
        if time != self.last_time and self.time_since_resampling > 10:
            #self.multinomial_resample()
            self.time_since_resampling += 1
            self.resampled = False
        else:
            self.time_since_resampling += 1
            self.resampled = False

        self.last_time = time

        for i, p in enumerate(self.particles):
            weights_update = p.update(spatial_measurement, feature_measurement, time, observation_id)
            print "Updating particle", i, " with weight: ", weights_update, ", particle weight: ", self.weights[i]
            self.weights[i] *= weights_update
        self.weights = 1./np.sum(self.weights)*self.weights

        print self.last_time
        print time

    def joint_update(self, spatial_measurements, feature_measurements, time, observation_id):

        if time != self.last_time:
            pass#

        #self.predict()

        #if self.last_time != time and self.last_time != -1:
        if time != self.last_time and self.time_since_resampling > 10:
            #self.multinomial_resample()
            self.time_since_resampling += 1
            self.resampled = False
        else:
            self.time_since_resampling += 1
            self.resampled = False

        self.last_time = time

        nbr_jumps = np.zeros((self.nbr_particles,))
        nbr_noise = np.zeros((self.nbr_particles,))
        nbr_assoc = np.zeros((self.nbr_particles,))
        for i, p in enumerate(self.particles):
            weights_update = self.particles[i].joint_update(spatial_measurements, feature_measurements, time, observation_id)
            print "Updating particle", i, " with weight: ", weights_update, ", particle weight: ", self.weights[i], ", did jump: ", p.did_jump, "nbr jumps: ", p.nbr_jumps
            self.weights[i] *= weights_update
            nbr_jumps[i] = p.nbr_jumps
            nbr_noise[i] = p.nbr_noise
            nbr_assoc[i] = p.nbr_assoc
        self.weights = 1./np.sum(self.weights)*self.weights

        #plt.plot(nbr_jumps, self.weights)
        plt.scatter(nbr_jumps, np.log(self.weights), marker="*", color='red')
        plt.scatter(nbr_noise, np.log(self.weights), marker="*", color='green')
        plt.scatter(nbr_assoc, np.log(self.weights), marker="*", color='blue')

        plt.xlabel('time (s)')
        plt.ylabel('voltage (mV)')
        plt.title('About as simple as it gets, folks')
        plt.grid(True)
        plt.savefig("test.png")
        plt.clf()
        plt.cla()
        #plt.show()

        if self.time_since_resampling > 4:
            self.multinomial_resample()
            #self.systematic_resample()
            pass
        else:
            self.time_since_resampling += 1

        print self.last_time
        print time

    def initialize_target(self, target_id, spatial_measurement, feature_measurement):

        for i, p in enumerate(self.particles):
            p.sm[target_id] = spatial_measurement
            p.fm[target_id] = feature_measurement
            p.sP[target_id] = self.spatial_std*self.spatial_std*np.eye(self.dim)
            p.fP[target_id] = self.feature_std*self.feature_std*np.eye(self.feature_dim)
