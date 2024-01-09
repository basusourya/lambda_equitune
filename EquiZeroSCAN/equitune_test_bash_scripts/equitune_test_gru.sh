#!/bin/bash

# to be run from the EquiSCAN folder

# seed 0 Add Jump
echo "seed 0 Add Jump"
jbsub -q x86_6h -cores 4+1 -require v100 python equitune_scan.py --use_attention --bidirectional --seed 0 --n_iters 10000 --learning_rate 0.00002 --save_dir "./equituned_models/" --split "add_jump" --equivariance "verb" --layer_type GRU --pretrained_model_dir "./models/add_jump/rnn_GRU_hidden_64_directions_2/seed_0/model0" --compute_test_accuracy

# seed 1 Add Jump
echo "seed 1 Add Jump"
jbsub -q x86_6h -cores 4+1 -require v100 python equitune_scan.py --use_attention --bidirectional --seed 1 --n_iters 10000 --learning_rate 0.00002 --save_dir "./equituned_models/" --split "add_jump" --equivariance "verb" --layer_type GRU --pretrained_model_dir "./models/add_jump/rnn_GRU_hidden_64_directions_2/seed_1/model0" --compute_test_accuracy

# seed 2 Add Jump
echo "seed 2 Add Jump"
jbsub -q x86_6h -cores 4+1 -require v100 python equitune_scan.py --use_attention --bidirectional --seed 2 --n_iters 10000 --learning_rate 0.00002 --save_dir "./equituned_models/" --split "add_jump" --equivariance "verb" --layer_type GRU --pretrained_model_dir "./models/add_jump/rnn_GRU_hidden_64_directions_2/seed_2/model0" --compute_test_accuracy

# seed 0 Around Right
echo "seed 0 Around Right"
jbsub -q x86_6h -cores 4+1 -require v100 python equitune_scan.py --use_attention --bidirectional --seed 0 --n_iters 10000 --learning_rate 0.00005 --save_dir "./equituned_models/" --split "around_right" --equivariance "direction" --layer_type GRU --pretrained_model_dir "./models/around_right/rnn_GRU_hidden_64_directions_2/seed_0/model0" --compute_test_accuracy

# seed 1 Around Right
echo "seed 1 Around Right"
jbsub -q x86_6h -cores 4+1 -require v100 python equitune_scan.py --use_attention --bidirectional --seed 1 --n_iters 10000 --learning_rate 0.00005 --save_dir "./equituned_models/" --split "around_right" --equivariance "direction" --layer_type GRU --pretrained_model_dir "./models/around_right/rnn_GRU_hidden_64_directions_2/seed_1/model0" --compute_test_accuracy

# seed 2 Around Right
echo "seed 2 Around Right"
jbsub -q x86_6h -cores 4+1 -require v100 python equitune_scan.py --use_attention --bidirectional --seed 2 --n_iters 10000 --learning_rate 0.00005 --save_dir "./equituned_models/" --split "around_right" --equivariance "direction" --layer_type GRU --pretrained_model_dir "./models/around_right/rnn_GRU_hidden_64_directions_2/seed_2/model0" --compute_test_accuracy
