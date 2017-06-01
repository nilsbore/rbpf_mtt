import numpy as np
import math
import copy
from cartesian import cartesian_inds
from rbpf_mtt.rbpf_particle import RBPFMParticle
import random
import matplotlib.pyplot as plt
import multiprocessing as mp
from functools import partial
import sys
#import pickle

def par_update_particle(spatial_measurements, feature_measurements, time, observation_id, location_ids, p):

    return p.update(spatial_measurements, feature_measurements, time, observation_id, location_ids), p

class RBPFMTTFilter(object):

    def __init__(self, nbr_targets, nbr_locations, nbr_particles, feature_dim, spatial_std=0.63, spatial_process_std=0.25,
                 feature_std=0.45, pjump=0.025, pnone=0.25, qjump=0.025, qnone=0.25):

        self.dim = 2 # I would like for this to be global instead
        self.feature_dim = feature_dim
        self.nbr_targets = nbr_targets
        self.nbr_locations = nbr_locations
        self.nbr_particles = nbr_particles
        self.spatial_process_std = spatial_process_std
        self.spatial_std = spatial_std
        self.feature_std = feature_std
        #self.feature_cov = feature_std*feature_std*np.identity(feature_dim)

        self.particles = [RBPFMParticle(self.dim, feature_dim, nbr_targets, nbr_locations, spatial_std, spatial_process_std,
                                        feature_std, pjump, pnone, qjump, qnone) for i in range(0, nbr_particles)]
        self.weights = 1./nbr_particles*np.ones((nbr_particles))

        self.last_time = -1

        self.resampled = False
        self.time_since_resampling = 0
        self.effective_sample_sizes = []

    def estimate(self):

        pos = np.zeros((self.nbr_targets, self.dim))
        feat = np.zeros((self.nbr_targets, self.feature_dim))
        feat_cov = np.zeros((self.nbr_targets, self.feature_dim, self.feature_dim))
        jumps = np.zeros((self.nbr_targets,))
        for i, p in enumerate(self.particles):
            pos += self.weights[i]*p.sm
            feat += self.weights[i]*p.fm
            feat_cov += self.weights[i]*p.fP
            jumps += self.weights[i]*p.target_jumps
            print p.c
        pos = 1./np.sum(self.weights)*pos # should already be normalized
        feat = 1./np.sum(self.weights)*feat # should already be normalized
        feat_cov = 1./np.sum(self.weights)*feat_cov # should already be normalized
        jumps = 1./np.sum(self.weights)*jumps

        return pos, feat, feat_cov, jumps

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

    def update(self, spatial_measurements, feature_measurements, time, observation_id, location_ids):

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

        #weight_updates = np.zeros((self.nbr_particles,))
        #for i, p in enumerate(self.particles):
        #    weight_updates[i] = self.particles[i].update(spatial_measurements, feature_measurements, time, observation_id, location_ids)

        func = partial(par_update_particle, spatial_measurements, feature_measurements, time, observation_id, location_ids)
        try:
            pool = mp.Pool(processes=4)
            results = pool.map(func, self.particles)
        finally:
            pool.close()
            pool.join()

        weight_updates, particle_updates = zip(*results)
        self.particles = list(particle_updates)

        self.weights *= np.array(weight_updates)
        self.weights = 1./np.sum(self.weights)*self.weights

        if self.effective_sample_size() < 0.5*float(self.nbr_particles):
            self.multinomial_resample()
            #self.systematic_resample()
        else:
            self.time_since_resampling += 1

        print self.last_time
        print time
        self.effective_sample_sizes.append(self.effective_sample_size())
        plt.scatter(np.arange(0, len(self.effective_sample_sizes)), np.array(self.effective_sample_sizes), marker="*", color='red')
        plt.savefig("samples.png")
        plt.clf()
        plt.cla()

        #particlestring = pickle.dumps(self.particles)
        #print "Number of particles: ", len(self.particles), " particle pickle length: ", len(particlestring)


    def initialize_target(self, target_id, spatial_measurement, feature_measurement, feature_covariance, location_id):

        for i, p in enumerate(self.particles):
            p.sm[target_id] = spatial_measurement
            p.fm[target_id] = feature_measurement
            p.sP[target_id] = self.spatial_std*self.spatial_std*np.eye(self.dim)
            p.fP[target_id] = 1.*feature_covariance #self.feature_std*self.feature_std*np.eye(self.feature_dim)
            p.c.append(target_id)
            p.location_ids[target_id] = location_id
            p.last_time = 0
            p.set_feature_cov(feature_covariance)
