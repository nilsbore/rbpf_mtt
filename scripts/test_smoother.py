#!/usr/bin/python

import numpy as np
import rospy
import actionlib
from rbpf_mtt.msg import GMMPoses, ObjectMeasurement
from rbpf_mtt.rbpf_filter import RBPFMTTFilter
#from rbpf_mtt.rbpf_smoother import RBPFMTTSmoother
from geometry_msgs.msg import PoseWithCovariance, Pose, PoseArray
from visualization_msgs.msg import Marker, MarkerArray
from rbpf_mtt.msg import ObservationDBAction, ObservationDBGoal, ObservationDBResult, ObservationDBFeedback
import os

class SmootherServer(object):

    _feedback = ObservationDBFeedback()
    _result   = ObservationDBResult()

    def __init__(self):

        self.nbr_targets = rospy.get_param('~number_targets', 2)
        self.feature_dim = rospy.get_param('~feature_dim', 2)

        max_iterations = 1000

        # N x 2 dimensions
        self.spatial_measurements = np.zeros((max_iterations, 2)) # can use the numpy library to save this
        # N x f dimensions
        self.feature_measurements = np.zeros((max_iterations, self.feature_dim)) # can use the numpy library to save this
        # N dimensions
        self.timesteps = np.zeros((max_iterations,), dtype=int)
        # N x 2 dimensions
        self.spatial_positions = np.zeros((max_iterations, self.nbr_targets, 2))
        # N dimensions
        self.target_ids = np.zeros((max_iterations,), dtype=int)
        # N dimensions, this is probably not really needed
        self.observation_ids = np.zeros((max_iterations,), dtype=int)

        self.target_poses = PoseArray()

        # We should probably save the feature and spatial noise used to generate the measurements
        # The easiest way to do this is to add parameters to the measurement_simulator node
        self.spatial_measurement_std = rospy.get_param('spatial_measurement_std', 0.1)
        self.feature_measurement_std = rospy.get_param('feature_measurement_std', 0.1)

        #self.gmm_pub = rospy.Publisher('filter_gmms', GMMPoses, queue_size=10)
        self.poses_pub = rospy.Publisher('set_target_poses', PoseArray, queue_size=10)
        self.obs_pub = rospy.Publisher('sim_filter_measurements', ObjectMeasurement, queue_size=10)

        #self.initialized = np.zeros((self.nbr_targets,), dtype=bool)

        self.is_playing = False
        self.iteration = 0
        rospy.Subscriber("filter_measurements", ObjectMeasurement, self.obs_callback)
        rospy.Subscriber("get_target_poses", PoseArray, self.poses_callback)

        self._action_name = "/observation_db"
        rospy.loginfo("Creating action server...")
        self._as = actionlib.SimpleActionServer(self._action_name, ObservationDBAction,
                                                execute_cb = self.execute_callback,
                                                auto_start = False)
        rospy.loginfo(" ...starting")
        self._as.start()
        rospy.loginfo(" ...done")

    def execute_callback(self, goal):

        SmootherServer._result.response = "Success!"
        #SmootherServer._result.success = True

        if goal.action == 'start':
            self.is_playing = False
            self.iteration = 0
        elif goal.action == 'stop':
            pass
        elif goal.action == 'save':
            self.save_observation_sequence(goal.observations_file)
        elif goal.action == 'load':
            self.load_observation_sequence(goal.observations_file)
            self.iteration = 0
        elif goal.action == 'replay':
            self.is_playing = True
            self.iteration = 0
        elif goal.action == 'step':
            self.step()
        else:
            SmootherServer._result.response = "Valid actions are: 'start', 'stop', 'save', 'load', 'replay', 'step'"
            #SmootherServer._result.success = False

        self._as.set_succeeded(SmootherServer._result)

    def step(self):

        if not self.is_playing:
            SmootherServer._result.response = "Can't step if not replay..."
            #SmootherServer._result.success = False
            return

        poses = PoseArray()
        poses.header.frame_id = 'map'
        for j in range(0, self.nbr_targets):
            p = Pose()
            p.position.x = self.spatial_positions[self.iteration, j, 0]
            p.position.y = self.spatial_positions[self.iteration, j, 1]
            poses.poses.append(p)

        self.poses_pub.publish(poses)

        obs = ObjectMeasurement()
        for i in range(0, self.feature_dim):
            obs.feature.append(self.feature_measurements[self.iteration, i])
        obs.pose.pose.position.x = self.spatial_measurements[self.iteration, 0]
        obs.pose.pose.position.y = self.spatial_measurements[self.iteration, 1]
        obs.initialization_id = self.target_ids[self.iteration]
        obs.observation_id = self.observation_ids[self.iteration]
        obs.timestep = self.timesteps[self.iteration]

        self.obs_pub.publish(obs)

        self.iteration += 1


    def save_observation_sequence(self, observations_file):

        self.spatial_measurements = self.spatial_measurements[:self.iteration,:]
        self.feature_measurements = self.feature_measurements[:self.iteration,:]
        self.timesteps = self.timesteps[:self.iteration]
        self.spatial_positions = self.spatial_positions[:self.iteration,:,:]
        self.target_ids = self.target_ids[:self.iteration]
        self.observation_ids = self.observation_ids[:self.iteration]

        np.savez(observations_file, spatial_measurements = self.spatial_measurements,
                                    feature_measurements = self.feature_measurements,
                                    timesteps = self.timesteps,
                                    spatial_positions = self.spatial_positions,
                                    target_ids = self.target_ids,
                                    observation_ids = self.observation_ids,
                                    spatial_measurement_std = self.spatial_measurement_std,
                                    feature_measurement_std = self.feature_measurement_std)

        SmootherServer._result.response = "Save observations at " + observations_file

    def load_observation_sequence(self, observations_file):
        if not os.path.isfile(observations_file):

            SmootherServer._result.response = "Could not load observations at " + observations_file

            return False
        npzfile = np.load(observations_file)
        self.spatial_measurements = npzfile['spatial_measurements']
        self.feature_measurements = npzfile['feature_measurements']
        self.timesteps = npzfile['timesteps']
        self.spatial_positions = npzfile['spatial_positions']
        self.target_ids = npzfile['target_ids']
        self.observation_ids = npzfile['observation_ids']
        self.spatial_measurement_std = npzfile['spatial_measurement_std']
        self.feature_measurement_std = npzfile['feature_measurement_std']

        SmootherServer._result.response = "Loaded observations at " + observations_file

        return True

    # here we got a measurement, with pose and feature, time is in the pose header
    def obs_callback(self, obs):

        if self.is_playing:
            return

        for i in range(0, self.feature_dim):
            self.feature_measurements[self.iteration, i] = obs.feature[i]
        self.spatial_measurements[self.iteration, 0] = obs.pose.pose.position.x
        self.spatial_measurements[self.iteration, 1] = obs.pose.pose.position.y
        self.target_ids[self.iteration] = obs.initialization_id
        self.observation_ids[self.iteration] = obs.observation_id
        self.timesteps[self.iteration] = obs.timestep

        for j, p in enumerate(self.target_poses.poses):
            self.spatial_positions[self.iteration, j, 0] = p.position.x
            self.spatial_positions[self.iteration, j, 1] = p.position.y

        self.iteration += 1

    def poses_callback(self, poses):

        self.target_poses = poses

    def publish_marginals(self, rbpfilter):

        gmms = filter_to_gmms(rbpfilter, self.initialized)
        for gmm in gmms:
            self.gmm_pub.publish(gmm)


if __name__ == '__main__':

    rospy.init_node('test_smoother', anonymous=True)

    ss = SmootherServer()

    rospy.spin()
