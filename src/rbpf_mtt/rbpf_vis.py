
import numpy as np
from geometry_msgs.msg import PoseWithCovariance
from rbpf_mtt.msg import GMMPoses

def filter_to_gmms(rbpfilter, initialized=None):

    if initialized is None:
        initialized=np.ones((rbpfilter.nbr_targets,), dtype=bool)

    gmms = []

    for j in range(0, rbpfilter.nbr_targets):
        if not initialized[j]:
            continue
        poses = GMMPoses()
        poses.id = j
        for i in range(0, rbpfilter.nbr_particles):
            poses.weights.append(rbpfilter.weights[i])
            m = rbpfilter.particles[i].sm[j]
            P = rbpfilter.particles[i].sP[j]
            pose = PoseWithCovariance()
            pose.pose.position.x = m[0]
            pose.pose.position.y = m[1]
            pose.pose.position.z = 0.
            # covariance is row-major
            pose.covariance[0] = P[0, 0]
            pose.covariance[1] = P[0, 1]
            pose.covariance[6] = P[1, 0]
            pose.covariance[7] = P[1, 1]
            poses.modes.append(pose)

        gmms.append(poses)

    return gmms

def particles_to_gmms(particles, weights, nbr_targets):

    gmms = []

    for j in range(0, nbr_targets):
        poses = GMMPoses()
        poses.id = j
        for i in range(0, len(particles)):
            poses.weights.append(weights[i])
            m = particles[i].sm[j]
            P = particles[i].sP[j]
            pose = PoseWithCovariance()
            pose.pose.position.x = m[0]
            pose.pose.position.y = m[1]
            pose.pose.position.z = 0.
            # covariance is row-major
            pose.covariance[0] = P[0, 0]
            pose.covariance[1] = P[0, 1]
            pose.covariance[6] = P[1, 0]
            pose.covariance[7] = P[1, 1]
            poses.modes.append(pose)

        gmms.append(poses)

    return gmms
