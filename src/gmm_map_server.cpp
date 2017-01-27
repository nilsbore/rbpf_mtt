#include <ros/ros.h>
#include <rbpf_mtt/GMMPoses.h>

using namespace std;

class GMMMapServer {
public:
    
    GMMMapServer()
    {

    }

    void subscriber(const rbpf_mtt::GMMPoses::ConstPtr& dist)
    {

    }

};

int main(int argc, char** argv)
{
    ros::init();

    return 0;
}
