# rbpf_mtt

Rao-Blackwellized Particle Filter for Multi-Target Jump Tracking of Object Detections

# Clean-up in progress!

I am currently in the process of cleaning up and documenting this code!
Please contact me if you are interested in using the code and I will try to aid.

# Setup

This package requires very little setup since it's mostly Python.
Two important dependencies are the [mongodb_store](https://github.com/strands-project/mongodb_store) and [SOMA](https://github.com/strands-project/soma) packages, which you can also install from
[the prebuilt STRANDS package repository](https://github.com/strands-project-releases/strands-releases/wiki).
To run though, you need to make sure that you have some data. For all kinds of scenarios
below, you need a 2d robot map generated from e.g. [gmapping](http://wiki.ros.org/gmapping).

# Running with pre-processed data

```
roslaunch rbpf_mtt track.launch map:=/path/to/map.yaml db_path:=/path/to/mongodb_db data_path:=/path/to/data number_targets:=7
```

# Running a simulation

1. Make sure you have a robot map in `/path/to/map.yaml`
2. Annotate the map with location regions using [the SOMA ROI manager](https://github.com/strands-project/soma#soma-roi-manager).
3. Start the launch file like before, make sure to specify the number of targets you want to track with the `number_targets` argument
4. Subscribe to the `/object_interactive_markers/update` interactive marker topic
5. Use the nav goal message in rviz to place the object targets one by one, the markers represent the "true" position of the objects
6. Drag the markers to move the objects around, click the blue boxes to make an observation of the corresponding location

# Processing meta room observations

See [the rbpf_processing package](https://github.com/nilsbore/rbpf_processing.git) for details.

# Benchmarking the method

In [the rbpf_benchmark package](https://github.com/nilsbore/rbpf_benchmark.git), I provide detailed
information on how to reproduce the results in the paper detailing this method.
