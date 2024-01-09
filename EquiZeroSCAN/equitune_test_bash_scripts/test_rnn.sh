#!/bin/bash

# to be run from the EquiSCAN folder

# seed 0
echo "Add Jump RNN seed 0"
python test_equivariant_scan.py --experiment_dir models/add_jump/rnn_RNN_hidden_64_directions_2/seed_0/model0/ --best_validation --layer_type RNN --split add_jump --equivariance none --use_attention --bidirectional --compute_test_accuracy --test_split add_jump

# seed 1
echo "Add Jump RNN seed 1"
python test_equivariant_scan.py --experiment_dir models/add_jump/rnn_RNN_hidden_64_directions_2/seed_1/model0/ --best_validation --layer_type RNN --split add_jump --equivariance none --use_attention --bidirectional --compute_test_accuracy --test_split add_jump

# seed 2
echo "Add Jump RNN seed 2"
python test_equivariant_scan.py --experiment_dir models/add_jump/rnn_RNN_hidden_64_directions_2/seed_2/model0/ --best_validation --layer_type RNN --split add_jump --equivariance none --use_attention --bidirectional --compute_test_accuracy --test_split add_jump

# seed 0
echo "Around Right RNN seed 0"
python test_equivariant_scan.py --experiment_dir models/around_right/rnn_RNN_hidden_64_directions_2/seed_0/model0/ --best_validation --layer_type RNN --split around_right --equivariance none --use_attention --bidirectional --compute_test_accuracy --test_split around_right

# seed 1
echo "Around Right RNN seed 1"
python test_equivariant_scan.py --experiment_dir models/around_right/rnn_RNN_hidden_64_directions_2/seed_1/model0/ --best_validation --layer_type RNN --split around_right --equivariance none --use_attention --bidirectional --compute_test_accuracy --test_split around_right

# seed 2
echo "Around Right RNN seed 2"
python test_equivariant_scan.py --experiment_dir models/around_right/rnn_RNN_hidden_64_directions_2/seed_2/model0/ --best_validation --layer_type RNN --split around_right --equivariance none --use_attention --bidirectional --compute_test_accuracy --test_split around_right
