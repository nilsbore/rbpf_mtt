import numpy as np
import math

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
        # the way this is supposed to work is that, when we sample a new c, we can only do it within one set

    def predict(self, measurement_partition=None):

        spatial_process_noise = 1.5
        feature_process_noise = 1.5

        sQ = np.identity(self.sm.shape[1]) # process noise
        fQ = np.identity(self.fm.shape[1]) # process noise

        for j in range(0, self.sm.shape[0]):
            self.sm[j] = self.sm[j]
            self.fm[j] = self.fm[j]
            self.sP[j] = self.sP[j] + sQ
            self.fP[j] = self.fP[j] + fQ

    # spatial_measurement is a vector with just one measurement of the object position
    # feature_measurement is a vector with a measurement of the object feature
    # time is a unique identifier > 0 for the measurement occasion
    def update(self, spatial_measurement, feature_measurement, time):

        #if len(self.c) == 0 or self.c[0] == 0:
        #    print "No associations sampled yet, can't update..."
        #    return

        if time != self.last_time:
            self.c = []
            self.last_time = time

        pclutter = 0.1 # probability of measurement originating from noise
        pdclutter = 0.1 # probability density of clutter measurement
        spatial_measurement_noise = 0.5
        feature_measurement_noise = 0.5

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

            likelihoods[j] = gauss_pdf(spatial_measurement, self.sm[j], sS) * \
                             gauss_pdf(feature_measurement, self.fm[j], fS)
        likelihoods[nbr_targets] = pdclutter


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

        pc = np.zeros((nbr_targets+1,))
        # probability of measurement given association, may arise from clutter anyways?
        pc[:nbr_targets] = (1.-pclutter)*likelihoods[:nbr_targets]
        pc[nbr_targets] = pclutter*likelihoods[nbr_targets]
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

        i = np.random.choice(nbr_targets+1, p=pc) #categ_rnd()
        # so what happens here if i ==
        weights_update = likelihoods[i]/pc[i]

        if i == nbr_targets:
            pass
            #self.c = nbr_targets # measurement is noise
            #we don't really care if it was associated with noise
        else:
            self.sm[i] = pot_sm[i]
            self.fm[i] = pot_fm[i]
            self.sP[i] = pot_sP[i]
            self.fP[i] = pot_fP[i]
            self.c.append(i)

        # Normalize the particles

        # Note: this should happen in the filter

        return weights_update



class RBPFMTTFilter(object):

    def __init__(self, nbr_targets, nbr_particles, feature_dim):

        self.dim = 2 # I would like for this to be global instead
        self.feature_dim = feature_dim
        self.nbr_targets = nbr_targets
        self.nbr_particles = nbr_particles

        self.particles = [RBPFMParticle(self.dim, feature_dim, nbr_targets) for i in range(0, nbr_particles)]
        self.weights = 1./nbr_particles*np.ones((nbr_particles))

        self.last_time = -1

        self.resampled = False

    def resample(self): # maybe this should take measurements?

        old_particles = self.particles

        samples = np.random.choice(self.nbr_particles, self.nbr_particles, p=self.weights)

        # now resample based on the weights
        for i in range(0, self.nbr_particles):
            self.weights[i] = 1./float(self.nbr_particles)
            self.particles[i] = old_particles[samples[i]]

        self.resampled = True

    # should this take a time interval also?
    def predict(self):

        for p in self.particles:
            p.predict()


    def single_update(self, spatial_measurement, feature_measurement, time):

        for i, p in enumerate(self.particles):
            weights_update = p.update(spatial_measurement, feature_measurement, time)
            print "Updating particle", i, " with weight: ", weights_update
            self.weights[i] *= weights_update
        self.weights = 1./np.sum(self.weights)*self.weights

        print self.last_time
        print time

        if self.last_time != time:
            self.last_time = time
            #self.resample()
            self.resampled = False
        else:
            self.resampled = False

    def update(self, spatial_measurements, feature_measurements, time=0.0):

        self.last_time += 1

        for m in range(0, spatial_measurements.shape[0]):
            for i, p in enumerate(self.particles):
                self.weights[i] *= p.single_update(spatial_measurements[m], feature_measurement[m], time)

    def initialize_target(self, target_id, spatial_measurement, feature_measurement):

        for i, p in enumerate(self.particles):
            p.sm[target_id] = spatial_measurement
            p.fm[target_id] = feature_measurement
            p.sP[target_id] = 1.0*np.eye(self.dim)
            p.fP[target_id] = 1.0*np.eye(self.feature_dim)


    #def estimate(self):

        # what do we want to estimate? basically just a list of objects
