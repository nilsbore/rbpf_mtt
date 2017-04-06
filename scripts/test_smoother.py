#!/usr/bin/python

import numpy as np
import rospy
from rbpf_mtt.msg import GMMPoses, ObjectMeasurement
from rbpf_mtt.srv import PublishGMMMap, PublishGMMMaps
from rbpf_mtt.rbpf_filter import RBPFMTTFilter
from rbpf_mtt.rbpf_smoother import RBPFMTTSmoother
from rbpf_mtt.rbpf_vis import filter_to_gmms, particles_to_gmms, estimates_to_markers, smoother_to_gmms, feature_estimates_to_poses
from geometry_msgs.msg import PoseWithCovariance
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Empty, Int32
from std_srvs.srv import Empty as EmptySrv
import sys

class SmootherNode(object):

    def __init__(self):

        self.gmm_pub = rospy.Publisher('filter_gmms', GMMPoses, queue_size=50)
        #self.poses_pub = rospy.Publisher('filter_poses', MarkerArray, queue_size=10)
        self.ready_pub = rospy.Publisher('filter_ready', Empty, queue_size=50)
        self.estimate_markers_pub = rospy.Publisher('estimate_markers', MarkerArray, queue_size=50)
        self.pose_estimates_pub = rospy.Publisher('pose_estimates', GMMPoses, queue_size=50)
        self.feature_estimates_pub = rospy.Publisher('feature_estimates', GMMPoses, queue_size=50)

        self.nbr_targets = rospy.get_param('~number_targets', 2)
        self.publish_maps = rospy.get_param('~publish_maps', True)
        self.spatial_std = rospy.get_param('~spatial_std', 0.63)
        self.feature_std = rospy.get_param('~feature_std', 0.45)
        self.feature_dim = rospy.get_param('~feature_dim', 4)
        self.number_particles = rospy.get_param('~number_particles', 100)

        self.pjump = rospy.get_param('~pjump', 0.025)
        self.pnone = rospy.get_param('~pnone', 0.25)
        self.qjump = rospy.get_param('~qjump', 0.025)
        self.qnone = rospy.get_param('~qnone', 0.25)

        self.smoother = RBPFMTTSmoother(self.nbr_targets, self.number_particles, self.feature_dim, 20, self.spatial_std,
                                        self.feature_std, self.pjump, self.pnone, self.qjump, self.qnone)
        self.initialized = np.zeros((self.nbr_targets,), dtype=bool)

        self.service = rospy.Service('smooth_estimates', EmptySrv, self.smooth_callback)

        self.last_time = -1
        self.last_observation_id = -1
        self.is_smoothed = False

        self.joint_spatial_measurement = None
        self.joint_feature_measurement = None
        self.location_ids = None
        self.all_timesteps = []
        self.split_timesteps = [[]]

        rospy.Subscriber("filter_measurements", ObjectMeasurement, self.callback)
        rospy.Subscriber("sim_filter_measurements", ObjectMeasurement, self.callback)
        rospy.Subscriber("smoother_vis", Int32, self.vis_callback)

    def measurements_from_pose(self, pose):
        measurement_dim = len(pose.feature)

        feature_measurement = np.zeros((measurement_dim,))
        spatial_measurement = np.zeros((2,))
        spatial_measurement[0] = pose.pose.pose.position.x
        spatial_measurement[1] = pose.pose.pose.position.y
        for i in range(0, measurement_dim):
            feature_measurement[i] = pose.feature[i]

        return spatial_measurement, feature_measurement, pose.location_id

    def covariance_from_pose(self, pose):

        measurement_dim = len(pose.feature)

        measurement_covariance = np.zeros((measurement_dim, measurement_dim))

        for i in range(0, measurement_dim):
            for j in range(0, measurement_dim):
                measurement_covariance[i, j] = pose.feature_covariance[i*measurement_dim+j]

        return measurement_covariance

    def publish_estimates(self):

        poses, feats, feat_covs, jumps = self.smoother.filter.estimate()

        feature_poses = feature_estimates_to_poses(feats, feat_covs, self.feature_dim, self.nbr_targets)

        markers = estimates_to_markers(poses, jumps)
        self.feature_estimates_pub.publish(feature_poses)
        self.estimate_markers_pub.publish(markers)

    def add_measurements(self):

        if np.all(self.initialized) and self.joint_spatial_measurement is not None:
            if self.joint_spatial_measurement.shape[0] > 4:
                print "Size: ", self.joint_spatial_measurement.shape[0]
                print "Exiting..."
                #print self.all_timesteps
                #print self.timesteps
                #sys.exit()
            self.smoother.update(self.joint_spatial_measurement,
                                 self.joint_feature_measurement,
                                 self.last_time,
                                 self.last_observation_id,
                                 self.locations_ids)
            if self.publish_maps:
                self.par_visualize_marginals(self.smoother.filter)
            self.publish_estimates()


    # here we got a measurement, with pose and feature, time is in the pose header
    def callback(self, pose):

        if self.is_smoothed:
            return

        self.measurement_covariance = self.covariance_from_pose(pose)

        self.all_timesteps.append(pose.timestep)
        is_init = np.all(self.initialized)
        spatial_measurement, feature_measurement, location_id = self.measurements_from_pose(pose)

        if self.joint_spatial_measurement is None or (is_init and pose.timestep != self.last_time):
            #self.last_time = pose.timestep
            #self.last_observation_id = pose.observation_id
            self.add_measurements()
            self.joint_spatial_measurement = None
            self.joint_feature_measurement = None
            self.joint_spatial_measurement = np.reshape(spatial_measurement, (1, -1))
            self.joint_feature_measurement = np.reshape(feature_measurement, (1, -1))
            self.timesteps = [ pose.timestep ]
            self.locations_ids = [ location_id ]
            self.timesteps.append(pose.timestep)
            self.split_timesteps.append([ pose.timestep ])
        elif is_init and pose.timestep != 0:
            self.joint_spatial_measurement = np.vstack((self.joint_spatial_measurement, spatial_measurement))
            self.joint_feature_measurement = np.vstack((self.joint_feature_measurement, feature_measurement))
            self.locations_ids = np.append(self.locations_ids, location_id)
            self.timesteps.append(pose.timestep)
            self.split_timesteps[-1].append(pose.timestep)

        self.last_time = pose.timestep
        self.last_observation_id = pose.observation_id

        print "time: ", pose.timestep
        #print "spatial measurement size: ", self.joint_spatial_measurement.shape

        print self.initialized
        print pose.initialization_id
        print self.nbr_targets
        if not is_init and pose.initialization_id != -1: #and pose.timestep == 0:
            print "Not intialized, adding initialization..."
            self.smoother.initialize_target(pose.initialization_id, spatial_measurement, feature_measurement,
                                            self.measurement_covariance, pose.timestep, location_id)
            self.initialized[pose.initialization_id] = True
            if np.all(self.initialized):
                self.par_visualize_marginals(self.smoother.filter)
                self.publish_estimates()

        else:
            if not is_init:
                print "All targets have not been initialized, not updating..."
                return
            print "Intialized, adding measurement..."
            #self.smoother.single_update(spatial_measurement, feature_measurement, pose.timestep, pose.observation_id)

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

    def par_visualize_smoothed_marginals(self, timestep):

        gmms = particles_to_gmms(self.smoother.timestep_particles[timestep],
            self.smoother.timestep_weights[timestep], self.nbr_targets)
        rospy.wait_for_service('publish_gmm_maps')
        try:
            publish_maps = rospy.ServiceProxy('publish_gmm_maps', PublishGMMMaps)
            publish_maps(gmms)
        except rospy.ServiceException, e:
            print "Service call failed: %s"%e

        smoothed_gmms = smoother_to_gmms(self.smoother, timestep)
        rospy.wait_for_service('smoother_publish_gmm_maps')
        try:
            publish_maps = rospy.ServiceProxy('smoother_publish_gmm_maps', PublishGMMMaps)
            publish_maps(smoothed_gmms)
        except rospy.ServiceException, e:
            print "Service call failed: %s"%e


    def smooth_callback(self, req):

        # add everything that hasn't been added
        self.add_measurements()

        self.smoother.smooth()

        self.is_smoothed = True

        print "Smoothed, all timesteps: ", self.all_timesteps
        print "Split timesteps: ", self.split_timesteps

        return ()

    def vis_callback(self, req):

        k = int(req.data)
        print "Timestep: ", k

        self.par_visualize_smoothed_marginals(k)

        e = Empty()
        self.ready_pub.publish(e)



if __name__ == '__main__':

    rospy.init_node('test_smoother', anonymous=True)

    sn = SmootherNode()

    rospy.spin()
