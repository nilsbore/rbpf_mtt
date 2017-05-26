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
        self.is_init = rospy.get_param('~is_init', True)
        self.data_path = rospy.get_param('~data_path', "")
        filter_location_clouds_string = rospy.get_param('~filter_location_clouds', "")
        self.filter_location_clouds = [int(i) for i in filter_location_clouds_string.split()]


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
        # N dimensions, location id
        self.location_ids = np.zeros((max_iterations,), dtype=int)

        # these are specifically for objects extracted through change detection
        self.cloud_paths = []
        self.central_images = []
        self.detection_type = []
        self.going_backward = []
        self.dims = []

        self.target_poses = PoseArray()

        # We should probably save the feature and spatial noise used to generate the measurements
        # The easiest way to do this is to add parameters to the measurement_simulator node
        self.spatial_measurement_std = rospy.get_param('spatial_measurement_std', 0.1)
        self.feature_measurement_std = rospy.get_param('feature_measurement_std', 0.1)
        self.measurement_covariance = self.feature_measurement_std*self.feature_measurement_std*np.identity(self.feature_dim)

        #self.gmm_pub = rospy.Publisher('filter_gmms', GMMPoses, queue_size=10)
        self.poses_pub = rospy.Publisher('set_target_poses', PoseArray, queue_size=50)
        self.obs_pub = rospy.Publisher('sim_filter_measurements', ObjectMeasurement, queue_size=50)
        self.smooth_pub = rospy.Publisher('smoother_vis', Int32, queue_size=50)
        self.done_pub = rospy.Publisher('playing_done', Empty, queue_size=50)
        self.labels_pub = rospy.Publisher('save_labels', Empty, queue_size=50)

        self.path_pubs = [rospy.Publisher('forward_detected_paths', String, queue_size=50),
                          rospy.Publisher('forward_propagated_paths', String, queue_size=50),
                          rospy.Publisher('backward_detected_paths', String, queue_size=50),
                          rospy.Publisher('backward_propagated_paths', String, queue_size=50)]
        self.init_path_pub = rospy.Publisher('init_paths', String, queue_size=50)
        self.init_paths = ""
        self.init_inds = -1*np.ones((self.nbr_targets,), dtype=int)
        self.init_poses = [Pose() for j in range(0, self.nbr_targets)]

        #self.initialized = np.zeros((self.nbr_targets,), dtype=bool)

        self.is_playing = False
        self.iteration = 0
        self.is_through = False
        self.autostep = False
        self.is_smoothed = False
        rospy.Subscriber("object_initialization_positions", ObjectMeasurement, self.init_callback)
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

    def object_type_index(self):

        if self.detection_type[self.iteration] == "detected":
            if self.going_backward[self.iteration]:
                return 2
            else:
                return 0
        elif self.detection_type[self.iteration] == "propagated":
            if self.going_backward[self.iteration]:
                return 3
            else:
                return 1
        else:
            print self.detection_type[self.iteration], " is not a proper detection type! Exiting..."
            sys.exit()

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

    def do_replay(self):

        self.is_playing = True
        self.iteration = 0
        SmootherServer._result.response = "Replaying observation sequence!"

        try:
            reset_filter = rospy.ServiceProxy('reset_filter', EmptySrv)
            reset_filter()
        except rospy.ServiceException, e:
            SmootherServer._result.response = "Is reset filter server running?"
            print "Service call failed: %s"%e

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
        elif goal.action == 'save_labels':
            self.labels_pub.publish()
            SmootherServer._result.response = "Sent message to save labels"
        elif goal.action == 'load':
            self.do_load(goal.observations_file)
        elif goal.action == 'replay':
            self.do_replay()
        elif goal.action == 'step':
            if not self.is_playing:
                self.do_load(goal.observations_file)
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
            SmootherServer._result.response = "Valid actions are: 'record', 'save', 'save_labels', 'load', 'replay', 'step', 'autostep'"
            #SmootherServer._result.success = False

        self._as.set_succeeded(SmootherServer._result)

    def step_callback(self, msg):

        print "Got step callback!"
        print "Iteration: ", self.iteration
        #print "Timesteps: ", self.timesteps
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
            self.done_pub.publish()
            #SmootherServer._result.success = False
            return

        # if self.is_smoothed:
        #     self.smooth_pub.publish(self.iteration)
        #     self.iteration += 1
        #     return

        first_timestep = self.timesteps[self.iteration]

        init_poses = PoseArray()
        init_poses.header.frame_id = 'map'
        init_poses.poses = self.init_poses
        init_inds = np.copy(self.init_inds)

        if len(self.cloud_paths) == 0:
            poses = PoseArray()
            poses.header.frame_id = 'map'
            for j in range(0, self.nbr_targets):
                p = Pose()
                p.position.x = self.spatial_positions[self.iteration, j, 0]
                p.position.y = self.spatial_positions[self.iteration, j, 1]
                poses.poses.append(p)

            self.poses_pub.publish(poses)

        clouds_paths = ["", "", "", ""]

        while True:

            obs = ObjectMeasurement()
            for i in range(0, self.feature_dim):
                obs.feature.append(self.feature_measurements[self.iteration, i])

            for i in range(0, self.feature_dim):
                for j in range(0, self.feature_dim):
                    obs.feature_covariance.append(self.measurement_covariance[i, j])

            obs.pose.pose.position.x = self.spatial_measurements[self.iteration, 0]
            obs.pose.pose.position.y = self.spatial_measurements[self.iteration, 1]
            obs.initialization_id = self.target_ids[self.iteration]
            obs.observation_id = self.observation_ids[self.iteration]
            obs.location_id = self.location_ids[self.iteration]
            print "Publishing observation: ", self.iteration, " with timestep: ", self.timesteps[self.iteration], " and location id", self.location_ids[self.iteration] #, " and cloud: ", self.cloud_paths[self.iteration]
            obs.timestep = self.timesteps[self.iteration]

            if len(self.cloud_paths) > 0:
                inds = np.where(init_inds == -1)[0]
                if len(inds) > 0:
                    init_poses.poses[inds[0]] = obs.pose.pose
                    init_inds[inds[0]] = 0 # dummy

            if len(self.cloud_paths) > 0:
                ind = self.object_type_index()
                if len(clouds_paths[ind]) > 0:
                    clouds_paths[ind] += ","
                clouds_paths[ind] += os.path.join(self.data_path, self.cloud_paths[self.iteration])

            if self.is_init:
                self.obs_pub.publish(obs)
            if self.is_smoothed:
                self.smooth_pub.publish(self.iteration)

            self.iteration += 1

            if (not self.step_by_timestep) or self.iteration >= len(self.timesteps) \
               or self.timesteps[self.iteration] != first_timestep:
               break

        if (self.iteration == 0 and self.is_init) or \
           (not self.is_init and len(self.cloud_paths) > 0):
            self.poses_pub.publish(init_poses)

        if len(self.filter_location_clouds) == 0 or self.location_ids[self.iteration] in self.filter_location_clouds:
            for i, paths in enumerate(clouds_paths):
                self.path_pubs[i].publish(paths)


    def save_observation_sequence(self, observations_file):

        observations_path = os.path.join(self.data_path, observations_file)

        if len(self.cloud_paths) == 0:
            self.spatial_measurements = self.spatial_measurements[:self.iteration,:]
            self.feature_measurements = self.feature_measurements[:self.iteration,:]
            self.timesteps = self.timesteps[:self.iteration]
            self.spatial_positions = self.spatial_positions[:self.iteration,:,:]
            self.target_ids = self.target_ids[:self.iteration]
            self.observation_ids = self.observation_ids[:self.iteration]
            self.location_ids = self.location_ids[:self.iteration]
            np.savez(observations_path, spatial_measurements = self.spatial_measurements,
                                        feature_measurements = self.feature_measurements,
                                        timesteps = self.timesteps,
                                        spatial_positions = self.spatial_positions,
                                        target_ids = self.target_ids,
                                        observation_ids = self.observation_ids,
                                        location_ids = self.location_ids,
                                        spatial_measurement_std = self.spatial_measurement_std,
                                        feature_measurement_std = self.feature_measurement_std,
                                        measurement_covariance = self.measurement_covariance)
        else:
            np.savez(observations_path, spatial_measurements = self.spatial_measurements,
                                        feature_measurements = self.feature_measurements,
                                        timesteps = self.timesteps,
                                        spatial_positions = self.spatial_positions,
                                        target_ids = self.target_ids,
                                        observation_ids = self.observation_ids,
                                        location_ids = self.location_ids,
                                        spatial_measurement_std = self.spatial_measurement_std,
                                        feature_measurement_std = self.feature_measurement_std,
                                        measurement_covariance = self.measurement_covariance,
                                        clouds = self.cloud_paths,
                                        central_images = self.central_images,
                                        detection_type = self.detection_type,
                                        going_backward = self.going_backward,
                                        dims = self.dims)

        SmootherServer._result.response = "Save observations at " + str(observations_path)

    def load_observation_sequence(self, observations_file):

        observations_path = os.path.join(self.data_path, observations_file)

        if not os.path.isfile(observations_path):

            SmootherServer._result.response = "Could not load observations at " + str(observations_path)

            return False
        npzfile = np.load(observations_path)
        self.spatial_measurements = npzfile['spatial_measurements']
        self.feature_measurements = npzfile['feature_measurements']
        self.timesteps = npzfile['timesteps']
        self.spatial_positions = npzfile['spatial_positions']
        self.target_ids = npzfile['target_ids']
        self.observation_ids = npzfile['observation_ids']
        self.location_ids = npzfile['location_ids']
        self.spatial_measurement_std = npzfile['spatial_measurement_std']
        self.feature_measurement_std = npzfile['feature_measurement_std']
        self.measurement_covariance = npzfile['measurement_covariance']
        if 'clouds' in npzfile:
            self.cloud_paths = npzfile['clouds']
            #self.is_init = False
        if 'detection_type' in npzfile:
            self.detection_type = npzfile['detection_type']
        if 'going_backward' in npzfile:
            self.going_backward = npzfile['going_backward']
        if 'central_images' in npzfile:
            self.central_images = npzfile['central_images']
        if 'dims' in npzfile:
            self.dims = npzfile['dims']

        # This is here if we have multiple points for the first timestep, which we don't if we picked them manually
        # inits = np.sum(self.timesteps == 0)
        # if inits > self.nbr_targets:
        #     indices = np.arange(self.nbr_targets, inits, dtype=int)
        #     self.spatial_measurements = np.delete(self.spatial_measurements, indices, axis=0)
        #     self.feature_measurements = np.delete(self.feature_measurements, indices, axis=0)
        #     self.timesteps = np.delete(self.timesteps, indices)
        #     self.spatial_positions = np.delete(self.spatial_positions, indices, axis=0)
        #     self.target_ids = np.delete(self.target_ids, indices)
        #     self.observation_ids = np.delete(self.observation_ids, indices)
        #     if len(self.cloud_paths) > 0:
        #         self.cloud_paths = list(self.cloud_paths[:self.nbr_targets]) + list(self.cloud_paths[inits:])
        #     if len(self.detection_type) > 0:
        #         self.detection_type = list(self.detection_type[:self.nbr_targets]) + list(self.detection_type[inits:])
        #     if len(self.going_backward) > 0:
        #         self.going_backward = np.delete(self.going_backward, indices)

        SmootherServer._result.response = "Loaded observations at " + observations_file
        print "Loaded observations sequence with timesteps: ", self.timesteps
        print "Init positions: ", self.spatial_measurements[:self.nbr_targets]
        print "Init features: ", self.feature_measurements[:self.nbr_targets]

        return True

    # initialization message, find the closes observation and publish its paths
    def init_callback(self, obs):

        # find the closest position in the previous timestep
        if self.iteration == 0:
            return

        last_timestep = self.timesteps[self.iteration-1]
        inds = np.where(self.timesteps == last_timestep)[0]

        pos = np.array([obs.pose.pose.position.x, obs.pose.pose.position.y])

        minval = 1000.0
        minind = -1
        for i in inds:
            vec = self.spatial_measurements[i, :2].flatten()#p.array([self.spatial_measurements[i, 0], self.spatial_measurements[i, 1]])
            val = np.linalg.norm(vec - pos)
            if val < minval:
                minval = val
                minind = i

        if minind == -1:
            return

        inds = np.where(self.init_inds == -1)[0]
        if len(inds) > 0:
            self.init_inds[inds[0]] = minind
            p = Pose()
            p.position.x = self.spatial_measurements[minind, 0]
            p.position.y = self.spatial_measurements[minind, 1]
            self.init_poses[inds[0]] = p

        if len(inds) == 1: # we initialized all the targets!
            self.spatial_measurements = np.vstack((self.spatial_measurements[self.init_inds, :], self.spatial_measurements))
            self.feature_measurements = np.vstack((self.feature_measurements[self.init_inds, :], self.feature_measurements))
            self.timesteps = np.concatenate((0*self.timesteps[self.init_inds], self.timesteps+1))
            self.spatial_positions = np.vstack((self.spatial_positions[inds, :], self.spatial_positions))
            self.target_ids = np.concatenate((np.arange(0, len(self.init_inds), dtype=int), self.target_ids))
            self.observation_ids = np.arange(0, len(self.timesteps), dtype=int)
            self.location_ids = np.concatenate((self.location_ids[self.init_inds], self.location_ids))
            self.cloud_paths = np.concatenate((self.cloud_paths[self.init_inds], self.cloud_paths))
            self.central_images = np.concatenate((self.central_images[self.init_inds], self.central_images))
            self.detection_type = np.concatenate((self.detection_type[self.init_inds], self.detection_type))
            self.going_backward = np.concatenate((self.going_backward[self.init_inds], self.going_backward))
            self.is_init = True
            self.is_playing = True
            self.iteration = 0

        if len(self.init_paths) > 0:
            self.init_paths += ","
        self.init_paths += os.path.join(self.data_path, self.cloud_paths[minind])

        self.init_path_pub.publish(self.init_paths)

        print "Minind: ", minind, ", min val: ", minval
        print "Closest cloud: ", self.cloud_paths[minind]
        print "Init inds: ", self.init_inds


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
        self.location_ids[self.iteration] = obs.location_id
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
