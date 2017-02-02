#!/usr/bin/python

import numpy as np
import rospy
from rbpf_mtt.msg import GMMPoses, ObjectMeasurement
#from rbpf_mtt.rbpf_filter import RBPFMTTFilter
from geometry_msgs.msg import Pose, PoseStamped, PoseWithCovariance
from visualization_msgs.msg import Marker, MarkerArray, InteractiveMarker, InteractiveMarkerControl
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from mongodb_store.message_store import MessageStoreProxy
from soma_msgs.msg import SOMAROIObject
from scipy.spatial import ConvexHull, Delaunay
#from interactive_markers.interactive_marker_server import *

def get_regions(db_name="somadata",collection="roi",roi_config="moving_objects"):
    msg_store = MessageStoreProxy(database=db_name,collection=collection)
    objs = msg_store.query(SOMAROIObject._type,message_query={"config":roi_config})
    regions,meta = zip(*objs)
    centers = np.zeros((len(regions), 2))
    for i, region in enumerate(regions):
        c = np.zeros((2,))
        for pose in region.posearray.poses:
            c[0] += pose.position.x
            c[1] += pose.position.y
        centers[i] = 1./float(len(region.posearray.poses))*c
    return regions, centers

class MeasurementVisServer(object):

    def __init__(self):

        self.marker_pub = rospy.Publisher('measurement_markers', MarkerArray, queue_size=10)
        self.positions_pub = rospy.Publisher('object_positions', ObjectMeasurement, queue_size=10)
        #self.object_pub = rospy.Publisher('measurement_markers', MarkerArray, queue_size=10)'

        self.nbr_targets = self.nbr_targets = rospy.get_param('~number_targets', 2)
        self.markers = MarkerArray()
        self.object_counters = np.zeros((self.nbr_targets,), dtype=int)
        self.initialized = np.zeros((self.nbr_targets,), dtype=bool)

        self.marker_server = InteractiveMarkerServer("object_interactive_markers")
        #self.room_server = InteractiveMarkerServer("room_interactive_markers")
        self.marker_poses = [Pose() for j in range(0, self.nbr_targets)]
        self.previous_poses = [Pose() for j in range(0, self.nbr_targets)]
        self.did_move = np.zeros((self.nbr_targets,), dtype=bool)
        self.marker_times = np.zeros((self.nbr_targets,), dtype=np.int64)

        self.regions, self.centers = get_regions()
        self.room_time = 0
        self.room_id = 0

        for i, region in enumerate(self.regions):
            print self.centers[i]

        #self.clear_markers()

        self.timestep = 0
        self.measurement_counter = 0

        #rospy.Timer(rospy.Duration(0.1), callback=self.maybe_publish_poses)
        rospy.Timer(rospy.Duration(0.1), callback=self.maybe_publish_rooms)
        rospy.Subscriber("filter_measurements", ObjectMeasurement, self.callback)

        self.initialize_room_markers()

    def maybe_publish_rooms(self, event):

        seconds = rospy.Time.now().to_sec()
        if self.room_time == 0 or seconds - self.room_time < 1:
            return

        room = self.regions[self.room_id]

        # ok, time to see if we have any objects within this region:
        vertices = np.zeros((len(room.posearray.poses), 2), dtype=float)
        for i, pose in enumerate(room.posearray.poses):
            vertices[i] = np.array([pose.position.x, pose.position.y])
        #hull = ConvexHull(vertices)
        #print vertices
        #if not isinstance(hull, Delaunay):
        #    hull = Delaunay(hull)
        hull = Delaunay(vertices)
        #hull = Delaunay(ConvexHull(vertices))

        published = False
        shuffled = np.arange(self.nbr_targets)
        np.random.shuffle(shuffled)
        for j in shuffled:#range(0, self.nbr_targets):
            if not self.initialized[j]:
                continue
            pose = [self.marker_poses[j].position.x, self.marker_poses[j].position.y]
            if hull.find_simplex(pose) >= 0:
                pose = PoseStamped()
                pose.header.frame_id = "map"
                pose.header.stamp = rospy.Time.now()
                pose.pose = self.marker_poses[j]
                object_pos = ObjectMeasurement()
                object_pos.pose = pose
                object_pos.initialization_id = j
                object_pos.timestep = self.timestep
                object_pos.observation_id = self.measurement_counter
                object_pos.negative_observation = False
                self.measurement_counter += 1
                self.positions_pub.publish(object_pos)
                self.room_time = 0
                published = True
            if self.did_move[j]:
                pose = [self.previous_poses[j].position.x, self.previous_poses[j].position.y]
                if hull.find_simplex(pose) >= 0:
                    print "Previous pose was inside, PUBLISHING!"
                    #self.did_move[j] = False # need to fix a history?
                    #self.previous_poses[j] = self.marker_poses[j]
                    pose = PoseStamped()
                    pose.header.frame_id = "map"
                    pose.header.stamp = rospy.Time.now()
                    pose.pose = self.previous_poses[j]
                    object_pos = ObjectMeasurement()
                    object_pos.pose = pose
                    object_pos.initialization_id = j
                    object_pos.timestep = self.timestep
                    object_pos.observation_id = self.measurement_counter
                    object_pos.negative_observation = True
                    self.positions_pub.publish(object_pos)
                    published = True

        if published:
            self.timestep = self.timestep + 1


    def maybe_publish_poses(self, event):

        seconds = rospy.Time.now().to_sec()
        for j in range(0, self.nbr_targets):
            mtime = self.marker_times[j]
            #if mtime != 0:
            #    print "Time diff for ", j , " is: ", mtime - seconds
            if mtime != 0 and seconds - mtime > 1:
                self.did_move[j] = True
                self.marker_times[j] = 0
                #pose = PoseStamped()
                #pose.header.frame_id = "map"
                #pose.header.stamp = rospy.Time.now()
                #pose.pose = self.marker_poses[j]
                #object_pos = ObjectMeasurement()
                #object_pos.pose = pose
                #object_pos.initialization_id = j
                #object_pos.observation_id = self.measurement_counter
                #self.measurement_counter += 1
                #self.positions_pub.publish(object_pos)
                #self.marker_times[j] = 0

    def object_id_color(self, object_id):

        colors =   {"vivid_yellow": (255, 179, 0),
                    "strong_purple": (128, 62, 117),
                    "vivid_orange": (255, 104, 0),
                    "very_light_blue": (166, 189, 215),
                    "vivid_red": (193, 0, 32),
                    "grayish_yellow": (206, 162, 98),
                    "medium_gray": (129, 112, 102),

                    # these aren't good for people with defective color vision:
                    "vivid_green": (0, 125, 52),
                    "strong_purplish_pink": (246, 118, 142),
                    "strong_blue": (0, 83, 138),
                    "strong_yellowish_pink": (255, 122, 92),
                    "strong_violet": (83, 55, 122),
                    "vivid_orange_yellow": (255, 142, 0),
                    "strong_purplish_red": (179, 40, 81),
                    "vivid_greenish_yellow": (244, 200, 0),
                    "strong_reddish_brown": (127, 24, 13),
                    "vivid_yellowish_green": (147, 170, 0),
                    "deep_yellowish_brown": (89, 51, 21),
                    "vivid_reddish_orange": (241, 58, 19),
                    "dark_olive_green": (35, 44, 22)}

        color = np.array(colors[colors.keys()[object_id]], dtype=float) / 255.0

        return color

    def clear_markers(self):

        clear_markers = MarkerArray()

        for i in range(0, 1000):
            clear_marker = Marker()
            clear_marker.header.frame_id = "map"
            clear_marker.header.stamp = rospy.Time.now()
            #clear_marker.type = Marker.SPHERE
            clear_marker.action = Marker.DELETE
            clear_marker.ns = "my_namespace"
            clear_marker.id = i
            #clear_marker.lifetime = rospy.Time(0)
            clear_markers.markers.append(clear_marker)

        self.marker_pub.publish(clear_markers)

    def initialize_room_markers(self):

        for room_id in range(0, len(self.regions)):

            pose = Pose()
            pose.position.x = self.centers[room_id, 0]
            pose.position.y = self.centers[room_id, 1]
            pose.position.z = 0.2
            pose.orientation.x = 0.
            pose.orientation.y = 0.
            pose.orientation.z = 0.
            pose.orientation.w = 1.

            marker = InteractiveMarker()
            marker.header.frame_id = "map"
            marker.name = "room_marker_" + str(room_id)
            marker.description = "Room " + str(room_id)

            # the marker in the middle
            box_marker = Marker()
            box_marker.type = Marker.CUBE
            box_marker.scale.x = 0.35
            box_marker.scale.y = 0.35
            box_marker.scale.z = 0.35
            box_marker.color.r = 0.
            box_marker.color.g = 0.
            box_marker.color.b = 1.
            box_marker.color.a = 1.
            box_marker.id = 1000

            # create a non-interactive control which contains the box
            box_control = InteractiveMarkerControl()
            box_control.always_visible = True
            #box_control.always_visible = False
            box_control.markers.append(box_marker)
            box_control.name = "button"
            box_control.interaction_mode = InteractiveMarkerControl.BUTTON
            marker.controls.append(box_control)
            #marker.controls.append(box_control)

            # move x
            #control = InteractiveMarkerControl()
            #control.orientation.w = 1
            #control.orientation.x = 0
            #control.orientation.y = 1
            #control.orientation.z = 0
            #control.always_visible = True
    #        control.name = "move_x"
    #        control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS


            self.marker_server.insert(marker, self.room_feedback)
            self.marker_server.applyChanges()
            self.marker_server.setPose( marker.name, pose )
            self.marker_server.applyChanges()

    def room_feedback(self, feedback):
        room_id = int(feedback.marker_name.rsplit('_', 1)[-1])
        print "Room id: ", room_id
        self.room_id = room_id
        self.room_time = rospy.Time.now().to_sec()

    def initialize_object_marker(self, object_id, pose):

        print "Adding interactive marker for object: ", object_id

        color = self.object_id_color(object_id)

        marker = InteractiveMarker()
        marker.header.frame_id = "map"
        marker.name = "object_marker_" + str(object_id)
        marker.description = "Object " + str(object_id)

        # the marker in the middle
        box_marker = Marker()
        box_marker.type = Marker.CUBE
        box_marker.scale.x = 0.25
        box_marker.scale.y = 0.25
        box_marker.scale.z = 0.25
        box_marker.color.r = color[0]
        box_marker.color.g = color[1]
        box_marker.color.b = color[2]
        box_marker.color.a = 1.
        box_marker.id = 1000

        # create a non-interactive control which contains the box
        box_control = InteractiveMarkerControl()
        box_control.always_visible = True
        #box_control.always_visible = False
        box_control.markers.append(box_marker)
        marker.controls.append(box_control)

        # move x
        control = InteractiveMarkerControl()
        control.orientation.w = 1
        control.orientation.x = 0
        control.orientation.y = 1
        control.orientation.z = 0
        control.always_visible = True
#        control.name = "move_x"
#        control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        control.name = "move_plane"
        control.interaction_mode = InteractiveMarkerControl.MOVE_PLANE
        marker.controls.append(control)

        self.marker_poses[object_id] = pose
        self.previous_poses[object_id] = pose
        self.marker_server.insert(marker, self.marker_feedback)
        self.marker_server.applyChanges()
        pose.position.z = 0.15
        self.marker_server.setPose( marker.name, pose )
        self.marker_server.applyChanges()

    def marker_feedback(self, feedback):
        #self.in_feedback=True
        #vertex_name = feedback.marker_name.rsplit('-', 1)
        object_id = int(feedback.marker_name.rsplit('_', 1)[-1])
        print "Marker id: ", object_id
        #self.topo_map.update_node_vertex(node_name, vertex_index, feedback.pose)
        #self.update_needed=True

        # just do something if there has been no updates for the
        # last x seconds
        feedback.pose.position.z = 0.15
        self.marker_poses[object_id] = feedback.pose
        self.marker_times[object_id] = rospy.Time.now().to_sec()


    def callback(self, clicked_pose):

        if not self.initialized[clicked_pose.initialization_id]:
            self.initialized[clicked_pose.initialization_id] = True
            self.initialize_object_marker(clicked_pose.initialization_id, clicked_pose.pose.pose)

        print "Got measurement, adding to GMMPoses ", clicked_pose.initialization_id

        color = self.object_id_color(clicked_pose.initialization_id)

        sphere_marker = Marker()
        sphere_marker.header.frame_id = "map"
        sphere_marker.header.stamp = rospy.Time.now()
        sphere_marker.ns = "my_namespace"
        sphere_marker.id = len(self.markers.markers)
        sphere_marker.type = Marker.SPHERE
        sphere_marker.action = Marker.ADD
        sphere_marker.pose.position.x = clicked_pose.pose.pose.position.x
        sphere_marker.pose.position.y = clicked_pose.pose.pose.position.y
        sphere_marker.pose.position.z = 0.2
        sphere_marker.pose.orientation.x = 0.
        sphere_marker.pose.orientation.y = 0.
        sphere_marker.pose.orientation.z = 0.
        sphere_marker.pose.orientation.w = 1.
        sphere_marker.scale.x = 0.2
        sphere_marker.scale.y = 0.2
        sphere_marker.scale.z = 0.2
        sphere_marker.color.a = 1. # Don't forget to set the alpha!
        sphere_marker.color.r = color[0]
        sphere_marker.color.g = color[1]
        sphere_marker.color.b = color[2]
        #sphere_marker.lifetime = rospy.Time(secs=1000)

        text_marker = Marker()
        text_marker.header.frame_id = "map"
        text_marker.header.stamp = rospy.Time.now()
        text_marker.ns = "my_namespace"
        text_marker.id = len(self.markers.markers)+1
        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.action = Marker.ADD
        text_marker.pose.position.x = clicked_pose.pose.pose.position.x
        text_marker.pose.position.y = clicked_pose.pose.pose.position.y
        text_marker.pose.position.z = 0.4
        text_marker.pose.orientation.x = 0.
        text_marker.pose.orientation.y = 0.
        text_marker.pose.orientation.z = 0.
        text_marker.pose.orientation.w = 1.
        text_marker.scale.z = 0.2
        text_marker.color.a = 1. # Don't forget to set the alpha!
        text_marker.color.r = 0.
        text_marker.color.g = 0.
        text_marker.color.b = 0.
        if clicked_pose.negative_observation:
            text_marker.text = "Negative " + str(self.object_counters[clicked_pose.initialization_id])
        else:
            text_marker.text = str(self.object_counters[clicked_pose.initialization_id])
        #text_marker.lifetime = rospy.Time(secs=1000)

        self.object_counters[clicked_pose.initialization_id] += 1

        self.markers.markers.append(sphere_marker)
        self.markers.markers.append(text_marker)

        self.marker_pub.publish(self.markers)

if __name__ == '__main__':

    rospy.init_node('measurement_vis_server', anonymous=True)

    mvs = MeasurementVisServer()

    rospy.spin()
