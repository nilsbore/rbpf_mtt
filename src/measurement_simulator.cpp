#include <ros/ros.h>
#include <geometry_msgs/Pose.h>
#include <eigen3/Eigen/Dense>
#include <rbpf_mtt/ObjectMeasurement.h>
#include <random>

using namespace std;

class MeasurementSimulator {
public:

    ros::NodeHandle n;
    ros::Subscriber clicked_sub;
    ros::Subscriber position_sub;
    ros::Publisher clicked_pub;
    bool initialized;
    int current_object_id;
    int timestep;
    int nbr_targets;
    int feature_dim;
    double feature_std;
    double spatial_std;
    Eigen::MatrixXd means;
    std::default_random_engine generator;

    MeasurementSimulator() : n()
    {
        ros::NodeHandle pn("~");
        pn.param<int>("number_targets", nbr_targets, 4);
        pn.param<int>("feature_dim", feature_dim, 4);
        pn.param<double>("feature_std", feature_std, 0.63);
        pn.param<double>("spatial_std", spatial_std, 0.45);

        std::default_random_engine generator;
        std::normal_distribution<double> feature_distribution(0.0, 1.0);

        means = Eigen::MatrixXd::Zero(feature_dim, nbr_targets);
        for (int j = 0; j < nbr_targets; ++j) {
            // so let's split this on a sphere around the origin
            Eigen::VectorXd feature = Eigen::VectorXd::Zero(feature_dim);
            for (int i = 0; i < feature_dim; ++i) {
                feature[i] = feature_distribution(generator);
            }
            feature.normalize();
            means.col(j) = 2.0*feature_std*feature; // this is just some random thingy
        }

        initialized = false;
        current_object_id = 0;
        timestep = 0;
        clicked_pub = n.advertise<rbpf_mtt::ObjectMeasurement>("filter_measurements", 50);
        clicked_sub = n.subscribe("/move_base_simple/goal", 50, &MeasurementSimulator::clicked_callback, this);
        position_sub = n.subscribe("object_positions", 50, &MeasurementSimulator::position_callback, this);

        cout << "Click to add initialization of object " << 0 << "..." << endl;
    }

    Eigen::VectorXd generate_feature(int object_id)
    {

        if (object_id == -1) {
            std::normal_distribution<double> feature_vec_distribution(0.0, 1.0);
            Eigen::VectorXd feature = Eigen::VectorXd::Zero(feature_dim);
            for (int i = 0; i < feature_dim; ++i) {
                feature[i] = feature_vec_distribution(generator);
            }
            feature.normalize();
            feature *= 2.0*feature_std;
            return feature;
        }

        // simply sample the feature from an N-dimensional multi_variate gaussian
        // let's assume that we have a few randomly generate mean vectors
        //std::default_random_engine generator;
        std::normal_distribution<double> feature_distribution(0.0, feature_std);
        Eigen::VectorXd feature = means.col(object_id);
        for (int i = 0; i < feature_dim; ++i) {
            feature(i) += feature_distribution(generator);
        }
        return feature;
    }

    Eigen::Vector2d generate_noise()
    {

        std::normal_distribution<double> spatial_distribution(0.0, spatial_std);
        Eigen::Vector2d noise;
        for (int i = 0; i < 2; ++i) {
            noise(i) += spatial_distribution(generator);
        }
        return noise;
    }

    void position_callback(const rbpf_mtt::ObjectMeasurement::ConstPtr& pos)
    {
        rbpf_mtt::ObjectMeasurement meas = *pos;

        Eigen::VectorXd feature = generate_feature(meas.initialization_id);
        Eigen::VectorXd noise = generate_noise();
        meas.pose.pose.position.x += noise(0);
        meas.pose.pose.position.y += noise(1);
        meas.feature.resize(feature_dim);
        for (int i = 0; i < feature_dim; ++i) {
            meas.feature[i] = feature[i];
        }
        clicked_pub.publish(meas);
    }

    void clicked_callback(const geometry_msgs::PoseStamped::ConstPtr& pose)
    {
        rbpf_mtt::ObjectMeasurement meas;
        meas.pose = *pose;

        if (initialized) {
            meas.initialization_id = current_object_id;
            meas.timestep = timestep;
        }
        else {
            meas.initialization_id = current_object_id;
        }

        if (current_object_id == nbr_targets - 1) {
            current_object_id = 0;
            initialized = true;
            timestep += 1;
        }
        else {
            current_object_id += 1;
        }

        if (initialized) {
            cout << "Click to add measurement of object " << current_object_id << "..." << endl;
        }
        else {
            cout << "Click to add initialization of object " << current_object_id << "..." << endl;
        }

        position_callback(rbpf_mtt::ObjectMeasurement::ConstPtr(new rbpf_mtt::ObjectMeasurement(meas)));
    }


};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "measurement_simulator");

    MeasurementSimulator ms;

    ros::spin();

    return 0;
}
