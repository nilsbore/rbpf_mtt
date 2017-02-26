#!/usr/bin/python

import numpy as np
import rospy
import actionlib
from rbpf_mtt.msg import GMMPoses, ObjectMeasurement
from rbpf_mtt.rbpf_filter import RBPFMTTFilter
#from rbpf_mtt.rbpf_smoother import RBPFMTTSmoother
from geometry_msgs.msg import PoseWithCovariance, Pose, PoseArray
from std_msgs.msg import Empty
from std_srvs.srv import Empty as EmptySrv
from std_msgs.msg import Int32
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray
from rbpf_mtt.msg import ObservationDBAction, ObservationDBGoal, ObservationDBResult, ObservationDBFeedback
import os

class SmootherServer(object):

    _feedback = ObservationDBFeedback()
    _result   = ObservationDBResult()

    def __init__(self):

        self.nbr_targets = rospy.get_param('~number_targets', 2)
        self.feature_dim = rospy.get_param('~feature_dim', 2)
        self.step_by_timestep = rospy.get_param('~step_by_timestep', True)

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

        self.cloud_paths = []

        self.target_poses = PoseArray()

        # We should probably save the feature and spatial noise used to generate the measurements
        # The easiest way to do this is to add parameters to the measurement_simulator node
        self.spatial_measurement_std = rospy.get_param('spatial_measurement_std', 0.1)
        self.feature_measurement_std = rospy.get_param('feature_measurement_std', 0.1)

        #self.gmm_pub = rospy.Publisher('filter_gmms', GMMPoses, queue_size=10)
        self.poses_pub = rospy.Publisher('set_target_poses', PoseArray, queue_size=10)
        self.obs_pub = rospy.Publisher('sim_filter_measurements', ObjectMeasurement, queue_size=10)
        self.smooth_pub = rospy.Publisher('smoother_vis', Int32, queue_size=10)
        self.path_pub = rospy.Publisher('cloud_paths', String, queue_size=10)

        #self.initialized = np.zeros((self.nbr_targets,), dtype=bool)

        self.is_playing = False
        self.iteration = 0
        self.is_through = False
        self.autostep = False
        self.is_smoothed = False
        rospy.Subscriber("filter_measurements", ObjectMeasurement, self.obs_callback)
        rospy.Subscriber("get_target_poses", PoseArray, self.poses_callback)

        # TEMP
        rospy.Subscriber("filter_ready", Empty, self.step_callback)

        self._action_name = "/observation_db"
        rospy.loginfo("Creating action server...")
        self._as = actionlib.SimpleActionServer(self._action_name, ObservationDBAction,
                                                execute_cb = self.execute_callback,
                                                auto_start = False)
        rospy.loginfo(" ...starting")
        self._as.start()
        rospy.loginfo(" ...done")

    def do_load(self, observations_file):
        self.load_observation_sequence(observations_file)
        self.is_smoothed = False
        self.is_through = False
        self.is_playing = True
        self.iteration = 0

    def do_autostep(self):
        SmootherServer._result.response = "Playing back!"
        self.autostep = True
        self.step()

    def do_smooth(self):
        SmootherServer._feedback.feedback = "Smoothing in progress..."
        SmootherServer._result.response = "Smoothing done!"
        try:
            smooth_estimates = rospy.ServiceProxy('smooth_estimates', EmptySrv)
            smooth_estimates()
        except rospy.ServiceException, e:
            SmootherServer._result.response = "Is smoothing server running?"
            print "Service call failed: %s"%e
        #self._as.action_server.publish_feedback()
        self.is_smoothed = True
        self.is_through = False
        self.iteration = 0
        self.is_playing = True

    def execute_callback(self, goal):

        SmootherServer._result.response = "Success!"
        #SmootherServer._result.success = True

        if goal.action == 'record':
            self.is_smoothed = False
            self.is_playing = False
            self.autostep = False
            self.iteration = 0
        elif goal.action == 'save':
            self.save_observation_sequence(goal.observations_file)
        elif goal.action == 'load':
            self.do_load(goal.observations_file)
        elif goal.action == 'replay':
            self.is_playing = True
            self.iteration = 0
        elif goal.action == 'step':
            self.autostep = False
            self.step()
        elif goal.action == 'autostep':
            self.do_autostep()
        elif goal.action == 'smooth':
            self.do_smooth()
        elif goal.action == 'autoload':
            self.do_load(goal.observations_file)
            self.do_autostep()
        #elif goal.action == 'autosmooth':
        #    self.do_load(goal.observations_file)
        #    self.do_autostep()
        #    self.do_smooth()
        else:
            SmootherServer._result.response = "Valid actions are: 'record', 'save', 'load', 'replay', 'step', 'autostep'"
            #SmootherServer._result.success = False

        self._as.set_succeeded(SmootherServer._result)

    def step_callback(self, msg):

        print "Got step callback!"
        print "Iteration: ", self.iteration
        print "Timesteps: ", self.timesteps
        print "Nbr timesteps: ", len(self.timesteps)
        if self.iteration < len(self.timesteps):
            print "Timestamp: ", self.timesteps[self.iteration]

        if not self.autostep:
            return

        self.step()

    def step(self):

        if not self.is_playing:
            SmootherServer._result.response = "Can't step if not replay..."
            #SmootherServer._result.success = False
            return

        if self.iteration >= len(self.timesteps):
            SmootherServer._result.response = "Done playing back, no more measurements!"
            self.is_through = True
            #SmootherServer._result.success = False
            return

        # if self.is_smoothed:
        #     self.smooth_pub.publish(self.iteration)
        #     self.iteration += 1
        #     return

        first_timestep = self.timesteps[self.iteration]

        poses = PoseArray()
        poses.header.frame_id = 'map'
        for j in range(0, self.nbr_targets):
            p = Pose()
            p.position.x = self.spatial_positions[self.iteration, j, 0]
            p.position.y = self.spatial_positions[self.iteration, j, 1]
            poses.poses.append(p)

        self.poses_pub.publish(poses)

        clouds_paths = ""

        while True:

            obs = ObjectMeasurement()
            for i in range(0, self.feature_dim):
                obs.feature.append(self.feature_measurements[self.iteration, i])
            obs.pose.pose.position.x = self.spatial_measurements[self.iteration, 0]
            obs.pose.pose.position.y = self.spatial_measurements[self.iteration, 1]
            obs.initialization_id = self.target_ids[self.iteration]
            obs.observation_id = self.observation_ids[self.iteration]
            obs.timestep = self.timesteps[self.iteration]

            if len(self.cloud_paths) > 0:
                if len(clouds_paths) > 0:
                    clouds_paths += ","
                clouds_paths += self.cloud_paths[self.iteration]

            self.obs_pub.publish(obs)
            if self.is_smoothed:
                self.smooth_pub.publish(self.iteration)

            self.iteration += 1

            if (not self.step_by_timestep) or self.iteration >= len(self.timesteps) \
               or self.timesteps[self.iteration] != first_timestep:
               break

        self.path_pub.publish(clouds_paths)


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
        if 'clouds' in npzfile:
            self.cloud_paths = npzfile['clouds']

        inits = np.sum(self.timesteps == 0)
        if inits > self.nbr_targets:
            indices = np.arange(self.nbr_targets, inits, dtype=int)
            self.spatial_measurements = np.delete(self.spatial_measurements, indices, axis=0)
            self.feature_measurements = np.delete(self.feature_measurements, indices, axis=0)
            self.timesteps = np.delete(self.timesteps, indices)
            self.spatial_positions = np.delete(self.spatial_positions, indices, axis=0)
            self.target_ids = np.delete(self.target_ids, indices)
            self.observation_ids = np.delete(self.observation_ids, indices)

        SmootherServer._result.response = "Loaded observations at " + observations_file
        print "Loaded observations sequence with timesteps: ", self.timesteps
        print "Init positions: ", self.spatial_measurements[:self.nbr_targets]
        print "Init features: ", self.feature_measurements[:self.nbr_targets]

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
