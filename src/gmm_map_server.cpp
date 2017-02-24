#include <ros/ros.h>
#include <rbpf_mtt/GMMPoses.h>
#include <nav_msgs/OccupancyGrid.h>
#include <costmap_2d/costmap_2d_ros.h>
#include <costmap_2d/costmap_2d_publisher.h>
#include <cmath>
#include <eigen3/Eigen/Dense>
#include <std_msgs/Empty.h>
#include <rbpf_mtt/PublishGMMMap.h>
#include <rbpf_mtt/PublishGMMMaps.h>
#include <geometry_msgs/PoseArray.h>

using namespace std;

double RationalApproximation(double t)
{
    // Abramowitz and Stegun formula 26.2.23.
    // The absolute value of the error should be less than 4.5 e-4.
    double c[] = {2.515517, 0.802853, 0.010328};
    double d[] = {1.432788, 0.189269, 0.001308};
    return t - ((c[2]*t + c[1])*t + c[0]) /
               (((d[2]*t + d[1])*t + d[0])*t + 1.0);
}

double NormalCDFInverse(double p)
{
    if (p <= 0.0 || p >= 1.0)
    {
        std::stringstream os;
        os << "Invalid input argument (" << p
           << "); must be larger than 0 but less than 1.";
        throw std::invalid_argument( os.str() );
    }

    // See article above for explanation of this section.
    if (p < 0.5)
    {
        // F^-1(p) = - G^-1(p)
        return -RationalApproximation( sqrt(-2.0*log(p)) );
    }
    else
    {
        // F^-1(p) = G^-1(1-p)
        return RationalApproximation( sqrt(-2.0*log(1-p)) );
    }
}

class GMMMapServer {
public:

    ros::NodeHandle n;
    ros::Subscriber sub;
    ros::ServiceServer service;
    ros::ServiceServer multi_service;
    int nbr_targets;
    vector<ros::Publisher> pubs;
    nav_msgs::OccupancyGrid map;
    vector<costmap_2d::Costmap2D> costmaps;
    vector<costmap_2d::Costmap2DPublisher*> costmap_publishers;
    ros::Publisher ready_pub;

    double map_height;
    double map_width;
    double map_origin_x;
    double map_origin_y;
    double map_res;

    string map_namespace;

    GMMMapServer() : n()
    {
        ros::NodeHandle pn("~");
        pn.param<int>("number_targets", nbr_targets, 4);
        pn.param<string>("namespace", map_namespace, "");

        pubs.resize(nbr_targets);
        for (int j = 0; j < nbr_targets; ++j) {
            pubs[j] = n.advertise<geometry_msgs::PoseArray>(map_namespace + "object_particles_" + to_string(j), 1, true); // latching
        }

        ready_pub = n.advertise<std_msgs::Empty>(map_namespace + "map_ready", 1);

        nav_msgs::OccupancyGrid::ConstPtr global_costmap_msg = ros::topic::waitForMessage<nav_msgs::OccupancyGrid>("/map", n, ros::Duration(5));
        if (!global_costmap_msg) {
            ROS_ERROR("Could not get global costmap, exiting...");
            exit(0);
        }
        map = *global_costmap_msg;

        ROS_INFO("DONE GETTING MAP");

        map_height = map.info.height;
        map_width = map.info.width;
        map_origin_x = map.info.origin.position.x;
        map_origin_y = map.info.origin.position.y;
        map_res = map.info.resolution;

        ROS_INFO_STREAM("Map width x height: " << map_width << " x " << map_height);

        costmaps.reserve(nbr_targets);
        costmap_publishers.reserve(nbr_targets);
        for (int j = 0; j < nbr_targets; ++j) {
            costmaps.push_back(costmap_2d::Costmap2D(map_width, map_height, map_res, map_origin_x, map_origin_y));
            //string topic_name = string("object_probabilities_") + to_string(j);
            //costmap_publishers.push_back(costmap_2d::Costmap2DPublisher(&n, &costmaps.back(), "/map", topic_name, true));
        }

        for (int j = 0; j < nbr_targets; ++j) {
            string topic_name = map_namespace + "object_probabilities_" + to_string(j);
            costmap_publishers.push_back(new costmap_2d::Costmap2DPublisher(&n, &costmaps[j], "/map", topic_name, true));
        }

        ROS_INFO("DONE INITIALIZING COSTMAPS");

        sub = n.subscribe(map_namespace + "filter_gmms", 1, &GMMMapServer::callback, this);
        service = n.advertiseService(map_namespace + "publish_gmm_map", &GMMMapServer::service_callback, this);//bpf_mtt::PublishGMMMap);
        multi_service = n.advertiseService(map_namespace + "publish_gmm_maps", &GMMMapServer::multi_service_callback, this);
    }

    /*
    double normalCDF(double value)
    {
       return 0.5 * erfc(-value / sqrt(2.0));
    }
    */

    void publish_map(const rbpf_mtt::GMMPoses& dist)
    {
        //costmap_2d::Costmap2D costmap(map_width, map_height, map_res, map_origin_x, map_origin_y);
        ROS_INFO("Got message callback!");
        /*if (!costmap_publishers[dist.id]->active()) {
            return;
        }*/

        Eigen::MatrixXd prob_accum = Eigen::MatrixXd::Zero(map_width, map_height);

        int map_x, map_y;
        for (size_t i = 0; i < dist.modes.size(); ++i) {

            Eigen::Matrix2d P;
            P << dist.modes[i].covariance[0], dist.modes[i].covariance[1],
                 dist.modes[i].covariance[6], dist.modes[i].covariance[7];

            double xstd = sqrt(P.col(0).norm());
            double ystd = sqrt(P.col(1).norm());

            double quantile = 2.0*NormalCDFInverse(0.95);
            double xmax = dist.modes[i].pose.position.x + quantile*xstd;
            double xmin = dist.modes[i].pose.position.x - quantile*xstd;
            double ymax = dist.modes[i].pose.position.y + quantile*ystd;
            double ymin = dist.modes[i].pose.position.y - quantile*ystd;

            //ROS_INFO_STREAM("res: " << map_res << ", xmax: " << xmax << ", xmin: " << xmin << ", ymax: " << ymax << ", ymin: " << ymin);

            // now we simply sample this at map.info.resolution / 2 resolution
            //double denom = 1.0/sqrt(2.0*M_PI*P.determinant());
            double denom = sqrt(2.0*M_PI*P.determinant());
            if (denom < 0.000001) {
                ROS_INFO_STREAM("Determinant very small, continuing...");
                continue;
            }
            //cout << "P: \n" << P << endl;
            Eigen::Matrix2d Pinv = P.inverse();
            //cout << "P^-1: \n" << Pinv << endl;

            int map_x, map_y;
            for (double y = ymin; y < ymax; y += map_res) {
                for (double x = xmin; x < xmax; x += map_res) {
                    Eigen::Vector2d v;
                    v << x - dist.modes[i].pose.position.x, y - dist.modes[i].pose.position.y;
                    costmaps[dist.id].worldToMapEnforceBounds(x, y, map_x, map_y);
                    if (map.data[costmaps[dist.id].getIndex(map_x, map_y)] != 0) {
                        prob_accum(map_x, map_y) = 0.0;
                        continue;
                    }
                    //prob_accum(map_y, map_x) += 1.0/denom*exp(-0.5*v.transpose()*Pinv*v);
                    //prob_accum(map_x, map_y) += exp(-0.5*v.transpose()*Pinv*v);
                    prob_accum(map_x, map_y) += 1.0/denom*exp(-0.5*v.transpose()*Pinv*v);
                    //ROS_INFO_STREAM("Acccum: " << prob_accum(map_y, map_x));
                }
            }

            // creating a window around this and sample the gaussian in this video is probably simplest
            // the problem is, how do we transform the gaussian covariance to this coordinate system?
            // an ugly way might be to do the worldToMap twice on the covariance transposed and not, hmm

            // ok, the strategy here will be to first discretize in world
            // coordinates and transform each pixel individually!


        }

        ROS_INFO("Done computing the accumulated map...");

        double maxprob = prob_accum.maxCoeff();
        prob_accum = 255.0/maxprob*prob_accum;

        ROS_INFO_STREAM("Max probability: " << maxprob);

        for (int i = 0; i < map_width; i++) {
            for (int j = 0; j < map_height; j++) {
                //costmap.setCost(i, j, uint8_t(prob_accum(i, j)));
                costmaps[dist.id].setCost(i, j, uint8_t(prob_accum(i, j)));
            }
        }

        ROS_INFO("Done inserting the accumulated map...");

        //string topic_name = string("object_probabilities_") + to_string(dist.id);
        //costmap_2d::Costmap2DPublisher cmp(&n, &costmap, "/map", topic_name, true);

        //cmp.publishCostmap();
        //pubs[dist.id].publish();

    }

    void publish_particles(const rbpf_mtt::GMMPoses& dist)
    {
        geometry_msgs::PoseArray particles;
        particles.header.frame_id = "/map";
        particles.header.stamp = ros::Time::now();

        particles.poses.resize(dist.modes.size());
        for (size_t i = 0; i < dist.modes.size(); ++i) {
            particles.poses[i] = dist.modes[i].pose;
        }

        pubs[dist.id].publish(particles);
    }

    void callback(const rbpf_mtt::GMMPoses::ConstPtr& dist)
    {
        publish_map(*dist);
        costmap_publishers[dist->id]->publishCostmap();
    }

    bool service_callback(rbpf_mtt::PublishGMMMap::Request& req, rbpf_mtt::PublishGMMMap::Response& res)
    {
        publish_map(req.gmm);
        costmap_publishers[req.gmm.id]->publishCostmap();
        return true;
    }

    bool multi_service_callback(rbpf_mtt::PublishGMMMaps::Request& req, rbpf_mtt::PublishGMMMaps::Response& res)
    {
        #pragma omp parallel for
        for (size_t i = 0; i < req.gmms.size(); ++i) {
            publish_map(req.gmms[i]);
        }

        for (size_t i = 0; i < req.gmms.size(); ++i) {
            publish_particles(req.gmms[i]);
            costmap_publishers[req.gmms[i].id]->publishCostmap();
        }
        return true;
    }

};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "gmm_map_server");

    GMMMapServer gms;

    ros::spin();

    return 0;
}
