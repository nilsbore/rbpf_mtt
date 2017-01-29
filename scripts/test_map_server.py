#!/usr/bin/python

import numpy as np
import rospy
from rbpf_mtt.msg import GMMPoses, ObjectMeasurement
#from rbpf_mtt.rbpf_filter import RBPFMTTFilter
from geometry_msgs.msg import PoseWithCovariance
from visualization_msgs.msg import Marker, MarkerArray

class TestMapServer(object):

    def __init__(self):

        self.gmm_pub = rospy.Publisher('filter_gmms', GMMPoses, queue_size=10)

        self.nbr_targets = self.nbr_targets = rospy.get_param('number_targets', 4)
        self.poses = [GMMPoses() for j in range(0, self.nbr_targets)]
        for j in range(0, self.nbr_targets):
            self.poses[j].id = j

        rospy.Subscriber("filter_measurements", ObjectMeasurement, self.callback)

    def callback(self, clicked_pose):

        print "Got measurement, adding to GMMPoses ", clicked_pose.initialization_id

        weight = 0.2
        pose = PoseWithCovariance()
        pose.pose = clicked_pose.pose.pose
        spatial_std = 1.0
        pose.covariance[0] = spatial_std*spatial_std
        pose.covariance[1] = 0.0
        pose.covariance[6] = 0.0
        pose.covariance[7] = spatial_std*spatial_std

        self.poses[clicked_pose.initialization_id].modes.append(pose)
        self.poses[clicked_pose.initialization_id].weights.append(weight)

        print "Publishing GMMPoses with length ", len(self.poses[clicked_pose.initialization_id].modes)

        self.gmm_pub.publish(self.poses[clicked_pose.initialization_id])


if __name__ == '__main__':

    rospy.init_node('test_map_server', anonymous=True)

    tms = TestMapServer()

    rospy.spin()
