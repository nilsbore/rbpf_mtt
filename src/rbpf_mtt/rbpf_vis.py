
import numpy as np
import rospy
from geometry_msgs.msg import PoseWithCovariance
from visualization_msgs.msg import Marker, MarkerArray
from rbpf_mtt.msg import GMMPoses

def object_id_color(object_id):

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

def estimates_to_markers(poses, jumps):

    estimate_markers = MarkerArray()

    nbr_targets = len(jumps)

    for j in range(0, nbr_targets):

        color = object_id_color(j)

        sphere_marker = Marker()
        sphere_marker.header.frame_id = "map"
        sphere_marker.header.stamp = rospy.Time.now()
        sphere_marker.ns = "estimate_namespace"
        sphere_marker.id = len(estimate_markers.markers)
        sphere_marker.type = Marker.CYLINDER
        sphere_marker.action = Marker.ADD
        sphere_marker.pose.position.x = poses[j, 0]
        sphere_marker.pose.position.y = poses[j, 1]
        sphere_marker.pose.position.z = 0.2
        sphere_marker.pose.orientation.x = 0.
        sphere_marker.pose.orientation.y = 0.
        sphere_marker.pose.orientation.z = 0.
        sphere_marker.pose.orientation.w = 1.
        sphere_marker.scale.x = 0.1
        sphere_marker.scale.y = 0.1
        sphere_marker.scale.z = 0.4
        sphere_marker.color.a = 1. # Don't forget to set the alpha!
        sphere_marker.color.r = color[0]
        sphere_marker.color.g = color[1]
        sphere_marker.color.b = color[2]
        #sphere_marker.lifetime = rospy.Time(secs=1000)

        estimate_markers.markers.append(sphere_marker)

        text_marker = Marker()
        text_marker.header.frame_id = "map"
        text_marker.header.stamp = rospy.Time.now()
        text_marker.ns = "estimate_namespace"
        text_marker.id = len(estimate_markers.markers)
        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.action = Marker.ADD
        text_marker.pose.position.x = poses[j, 0]
        text_marker.pose.position.y = poses[j, 1]
        text_marker.pose.position.z = 0.5
        text_marker.pose.orientation.x = 0.
        text_marker.pose.orientation.y = 0.
        text_marker.pose.orientation.z = 0.
        text_marker.pose.orientation.w = 1.
        text_marker.scale.z = 0.2
        text_marker.color.a = 1. # Don't forget to set the alpha!
        text_marker.color.r = 0.
        text_marker.color.g = 0.
        text_marker.color.b = 0.
        text_marker.text = "Jumps: " + str(int(round(jumps[j])))

        estimate_markers.markers.append(text_marker)

    return estimate_markers
