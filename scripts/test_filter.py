#!/usr/bin/python

import numpy as np
import rospy
from rbpf_mtt.msg import GMMPoses, ObjectMeasurement
from rbpf_mtt.srv import PublishGMMMap
from rbpf_mtt.rbpf_filter import RBPFMTTFilter
from rbpf_mtt.rbpf_vis import filter_to_gmms
from geometry_msgs.msg import PoseWithCovariance
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Empty

class FilterServer(object):

    def __init__(self):

        self.gmm_pub = rospy.Publisher('filter_gmms', GMMPoses, queue_size=10)
        self.poses_pub = rospy.Publisher('filter_poses', MarkerArray, queue_size=10)
        self.ready_pub = rospy.Publisher('filter_ready', Empty, queue_size=10)

        self.nbr_targets = rospy.get_param('~number_targets', 2)

        self.filter = RBPFMTTFilter(self.nbr_targets, 100, 4)
        self.initialized = np.zeros((self.nbr_targets,), dtype=bool)

        rospy.Subscriber("filter_measurements", ObjectMeasurement, self.callback)
        rospy.Subscriber("sim_filter_measurements", ObjectMeasurement, self.callback)

    # here we got a measurement, with pose and feature, time is in the pose header
    def callback(self, pose):

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
            self.filter.initialize_target(pose.initialization_id, spatial_measurement, feature_measurement)
            self.initialized[pose.initialization_id] = True
        else:
            if not is_init:
                print "All targets have not been initialized, not updating..."
                return
            print "Intialized, adding measurement..."
            #if pose.negative_observation:
            #    self.filter.negative_update(spatial_measurement, pose.observation_id)
            #else:
            #    self.filter.single_update(spatial_measurement, feature_measurement, pose.timestep, pose.observation_id)
            self.filter.single_update(spatial_measurement, feature_measurement, pose.timestep, pose.observation_id)

        self.visualize_marginals(self.filter)
        self.publish_marginals(self.filter)
        e = Empty()
        self.ready_pub.publish(e)

    def publish_marginals(self, rbpfilter):

        gmms = filter_to_gmms(rbpfilter, self.initialized)
        for gmm in gmms:
            self.gmm_pub.publish(gmm)

    def visualize_marginals(self, rbpfilter):

        gmms = filter_to_gmms(rbpfilter, self.initialized)
        rospy.wait_for_service('publish_gmm_map')
        for gmm in gmms:
            try:
                publish_map = rospy.ServiceProxy('publish_gmm_map', PublishGMMMap)
                publish_map(gmm)
            except rospy.ServiceException, e:
                print "Service call failed: %s"%e

    # This should just publish a posearray, which can be displayed directly
    # But how do we know which pose is which? Maybe it would make more
    # Sense with a marker array with a label for each marker
    def publish_estimated_poses(self, rbpfilter):
        markers = MarkerArray()

        poses = self.filter.estimate()

        for j in range(0, poses.shape[0]):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "my_namespace"
            marker.id = j
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = poses[j, 0]
            marker.pose.position.y = poses[j, 1]
            marker.pose.position.z = 0.
            marker.pose.orientation.x = 0.
            marker.pose.orientation.y = 0.
            marker.pose.orientation.z = 0.
            marker.pose.orientation.w = 1.
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2
            marker.color.a = 1. # Don't forget to set the alpha!
            marker.color.r = 0.
            marker.color.g = 1.
            marker.color.b = 0.
            markers.markers.append(marker)

        self.poses_pub.publish(markers)


if __name__ == '__main__':

    rospy.init_node('test_filter', anonymous=True)

    fs = FilterServer()

    rospy.spin()
