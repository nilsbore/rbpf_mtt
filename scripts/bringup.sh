SESSION=simulation
# This is the workspace containing the ros packages that are needed
# For this experiment, the repos rcv_bringup, rcv_common, rcv_description,
# rcv_slam and rcv_localization should be enough
WS_PATH="/home/nbore/Workspace/ros/catkin_ws"
DB_PATH="/home/nbore/moving_objects_db"
DATA_PATH="/media/nbore/data_ssd/moving_chairs_semantic_map"
SOURCE_WS="${WS_PATH}/devel/setup.bash"
# Built with the rcv_fuser_node from rcv_slam

tmux -2 new-session -d -s $SESSION

tmux new-window -t $SESSION:0 -n 'roscore'
#tmux new-window -t $SESSION:1 -n 'mongodb'
#tmux new-window -t $SESSION:2 -n 'soma'
tmux new-window -t $SESSION:1 -n 'rviz'
tmux new-window -t $SESSION:2 -n 'track'
tmux new-window -t $SESSION:3 -n 'axclient'
tmux new-window -t $SESSION:4 -n 'benchmark'
tmux new-window -t $SESSION:5 -n 'annotate'
tmux new-window -t $SESSION:6 -n 'feature_vis'

tmux select-window -t $SESSION:0
tmux send-keys "roscore" C-m

tmux select-window -t $SESSION:1
tmux send-keys "source $SOURCE_WS" C-m
tmux send-keys "rosrun rviz rviz"

tmux select-window -t $SESSION:2
tmux send-keys "source $SOURCE_WS" C-m
#tmux send-keys "roslaunch quasimodo_retrieval retrieval.launch vocabulary_path:=/home/nbore/Data/tsc_semantic_maps/vocabulary"
tmux send-keys "roslaunch rbpf_mtt track.launch map:=${WS_PATH}/src/rbpf_mtt/maps/dynamic_map.yaml db_path:=${DB_PATH} number_targets:=7 data_path:=/media/nbore/data_ssd/processed_data/moving_chairs_semantic_map"

tmux select-window -t $SESSION:3
tmux send-keys "source $SOURCE_WS" C-m
tmux send-keys "rosrun actionlib axclient.py /observation_db"

tmux select-window -t $SESSION:4
tmux send-keys "source $SOURCE_WS" C-m
tmux send-keys "roslaunch rbpf_benchmark benchmark.launch number_targets:=7 data_path:=/media/nbore/data_ssd/processed_data/moving_chairs_semantic_map"

tmux select-window -t $SESSION:5
tmux send-keys "source $SOURCE_WS" C-m
tmux send-keys "roslaunch rbpf_benchmark annotate.launch map:=${WS_PATH}/src/rbpf_mtt/maps/dynamic_map.yaml number_targets:=7 data_path:=/media/nbore/data_ssd/processed_data/moving_chairs_semantic_map"

tmux select-window -t $SESSION:6
tmux send-keys "source $SOURCE_WS" C-m
tmux send-keys "rosrun rbpf_processing visualize_features.py"

# Set default window
tmux select-window -t $SESSION:0

# Attach to session
tmux -2 attach-session -t $SESSION
