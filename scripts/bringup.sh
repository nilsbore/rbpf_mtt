SESSION=simulation
# This is the workspace containing the ros packages that are needed
# For this experiment, the repos rcv_bringup, rcv_common, rcv_description,
# rcv_slam and rcv_localization should be enough
SOURCE_WS="/home/nbore/instance_places/catkin_ws/devel/setup.bash"
# Built with the rcv_fuser_node from rcv_slam

tmux -2 new-session -d -s $SESSION

tmux new-window -t $SESSION:0 -n 'roscore'
#tmux new-window -t $SESSION:1 -n 'mongodb'
#tmux new-window -t $SESSION:2 -n 'soma'
tmux new-window -t $SESSION:1 -n 'rviz'
tmux new-window -t $SESSION:2 -n 'simulation'
#tmux new-window -t $SESSION:5 -n 'create_query_models'
#tmux new-window -t $SESSION:6 -n 'graph_node'
#tmux new-window -t $SESSION:7 -n 'inference_node'

tmux select-window -t $SESSION:0
tmux send-keys "roscore" C-m

tmux select-window -t $SESSION:1
tmux send-keys "source $SOURCE_WS" C-m
tmux send-keys "rosrun rviz rviz"

tmux select-window -t $SESSION:2
tmux send-keys "source $SOURCE_WS" C-m
#tmux send-keys "roslaunch quasimodo_retrieval retrieval.launch vocabulary_path:=/home/nbore/Data/tsc_semantic_maps/vocabulary"
tmux send-keys "roslaunch rbpf_mtt test.launch map:=/home/nbore/dynamic_map.yaml db_path:=/home/nbore/moving_objects_db number_targets:=2"

# Set default window
tmux select-window -t $SESSION:0

# Attach to session
tmux -2 attach-session -t $SESSION