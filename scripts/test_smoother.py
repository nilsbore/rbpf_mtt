#!/usr/bin/python

import numpy as np
import rospy
from rbpf_mtt.msg import GMMPoses, ObjectMeasurement
from rbpf_mtt.srv import PublishGMMMap, PublishGMMMaps
from rbpf_mtt.rbpf_filter import RBPFMTTFilter
from rbpf_mtt.rbpf_smoother import RBPFMTTSmoother
from rbpf_mtt.rbpf_vis import filter_to_gmms, particles_to_gmms
from geometry_msgs.msg import PoseWithCovariance
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Empty, Int32
from std_srvs.srv import Empty as EmptySrv

class SmootherNode(object):

    def __init__(self):

        self.gmm_pub = rospy.Publisher('filter_gmms', GMMPoses, queue_size=10)
        #self.poses_pub = rospy.Publisher('filter_poses', MarkerArray, queue_size=10)
        self.ready_pub = rospy.Publisher('filter_ready', Empty, queue_size=10)

        self.nbr_targets = rospy.get_param('~number_targets', 2)
        self.publish_maps = rospy.get_param('~publish_maps', True)

        self.smoother = RBPFMTTSmoother(self.nbr_targets, 100, 4, 10)
        self.initialized = np.zeros((self.nbr_targets,), dtype=bool)

        self.service = rospy.Service('smooth_estimates', EmptySrv, self.smooth_callback)

        self.last_time = -1
        self.is_smoothed = False

        rospy.Subscriber("filter_measurements", ObjectMeasurement, self.callback)
        rospy.Subscriber("sim_filter_measurements", ObjectMeasurement, self.callback)
        rospy.Subscriber("smoother_vis", Int32, self.vis_callback)

    # here we got a measurement, with pose and feature, time is in the pose header
    def callback(self, pose):

        if self.is_smoothed:
            return

        if pose.timestep != self.last_time and self.publish_maps:
            self.last_time = pose.timestep
            self.par_visualize_marginals(self.smoother.filter)

        measurement_dim = len(pose.feature)

        feature_measurement = np.zeros((measurement_dim,))
        spatial_measurement = np.zeros((2,))
        spatial_measurement[0] = pose.pose.pose.position.x
        spatial_measurement[1] = pose.pose.pose.position.y
        for i in range(0, measurement_dim):
            feature_measurement[i] = pose.feature[i]

        is_init = np.all(self.initialized)
        print self.initialized
        print pose.initialization_id
        print self.nbr_targets
        if not is_init and pose.initialization_id != -1:
            print "Not intialized, adding initialization..."
            self.smoother.initialize_target(pose.initialization_id, spatial_measurement, feature_measurement, pose.timestep)
            self.initialized[pose.initialization_id] = True
        else:
            if not is_init:
                print "All targets have not been initialized, not updating..."
                return
            print "Intialized, adding measurement..."
            self.smoother.single_update(spatial_measurement, feature_measurement, pose.timestep, pose.observation_id)

        # We should add an argument to only do this in some cases
        #self.par_visualize_marginals(self.smoother.filter)
        e = Empty()
        self.ready_pub.publish(e)

    def par_visualize_marginals(self, rbpfilter):

        gmms = filter_to_gmms(rbpfilter, self.initialized)
        rospy.wait_for_service('publish_gmm_maps')
        try:
            publish_maps = rospy.ServiceProxy('publish_gmm_maps', PublishGMMMaps)
            publish_maps(gmms)
        except rospy.ServiceException, e:
            print "Service call failed: %s"%e

    def par_visualize_smoothed_marginals(self, particles, weights):

        gmms = particles_to_gmms(particles, weights, self.nbr_targets)
        rospy.wait_for_service('publish_gmm_maps')
        try:
            publish_maps = rospy.ServiceProxy('publish_gmm_maps', PublishGMMMaps)
            publish_maps(gmms)
        except rospy.ServiceException, e:
            print "Service call failed: %s"%e


    def smooth_callback(self, req):

        self.smoother.smooth()

        self.is_smoothed = True

        return ()

    def vis_callback(self, req):

        k = int(req.data)
        print "Timestep: ", k

        self.par_visualize_smoothed_marginals(self.smoother.timestep_particles[k],
                                              self.smoother.timestep_weights[k])

        e = Empty()
        self.ready_pub.publish(e)



if __name__ == '__main__':

    rospy.init_node('test_smoother', anonymous=True)

    sn = SmootherNode()

    rospy.spin()