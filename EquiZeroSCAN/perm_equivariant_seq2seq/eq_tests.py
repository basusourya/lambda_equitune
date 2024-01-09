import random
import argparse
import os

import torch
import torch.nn as nn

import utils as utils
from perm_equivariant_seq2seq.models import BasicSeq2Seq
from perm_equivariant_seq2seq.equitune_models import EquiSCANModel
from perm_equivariant_seq2seq.equizero_models import EquiZeroSCANModel
from perm_equivariant_seq2seq.data_utils import get_scan_split
from perm_equivariant_seq2seq.utils import tensors_from_pair
from perm_equivariant_seq2seq.data_utils import get_invariant_scan_languages
from perm_equivariant_seq2seq.g_utils import cyclic_group_generator, cyclic_group, g_transform_data, g_inv_transform_prob_data, g_transform_prob_data


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def equizerotest(input_tensor,
          target_tensor,
          model,
          in_G,
          out_G):
    """Perform one training iteration for the model

    Args:
        input_tensor: (torch.tensor) Tensor representation (1-hot) of sentence
        in input language
        target_tensor: (torch.tensor) Tensor representation (1-hot) of target
        sentence in output language
        model_to_train: (nn.Module: Seq2SeqModel) seq2seq model being trained
        in_G: input group
        out_G: output_group
        enc_optimizer: (torch.optimizer) Optimizer object for model encoder
        dec_optimizer: (torch.optimizer) Optimizer object for model decoder
        loss_fn: (torch.nn.Loss) Loss object used for training
        teacher_forcing_ratio: (float) Ratio with which true word is used as
        input to decoder
    Returns:
        (torch.scalar) Value of loss achieved by model at current iteration
    """
    model.eval()

    # input eq word indices [4, 8]
    transformed_input_tensor = g_transform_data(input_tensor, in_G, device)
    # generate transformed target_tensors only for teacher forcing
    # output eq word indices [2, 5]
    transformed_target_tensor = g_transform_data(target_tensor, out_G, device)
    # example model_output
    # -2.0257, -1.9021, -1.9692, -2.1580, -2.1741, -1.9069, -2.0696, -2.6039],
    # [-2.1432, -2.2621, -2.0530, -1.9556, -2.0481, -2.0793, -2.0876, -2.0397],
    # [-2.0652, -2.0721, -2.0830, -2.0721, -2.0667, -2.0631, -2.1637, -2.0541],
    model_output = model(input_tensor=input_tensor,
                                  target_tensor=target_tensor)
    # example transformed_model_output
    # [-2.0257, -1.9021, -1.9069, -2.1580, -2.1741, -1.9692, -2.0696, -2.6039],
    # [-2.1432, -2.2621, -2.0793, -1.9556, -2.0481, -2.0530, -2.0876, -2.0397],
    # [-2.0652, -2.0721, -2.0631, -2.0721, -2.0667, -2.0830, -2.1637, -2.0541],
    transformed_model_output = g_transform_prob_data(model_output, out_G)
    # example model_transformed_output
    # [-2.0257, -1.9021, -1.9069, -2.1580, -2.1741, -1.9692, -2.0696, -2.6039],
    # [-2.1432, -2.2621, -2.0793, -1.9556, -2.0481, -2.0530, -2.0876, -2.0397],
    # [-2.0652, -2.0721, -2.0631, -2.0721, -2.0667, -2.0830, -2.1637, -2.0541],
    model_transformed_output = model(input_tensor=transformed_input_tensor[1],
                                  target_tensor=transformed_target_tensor[1])

    assert torch.allclose(transformed_model_output, model_transformed_output)


def equitest(input_tensor,
          target_tensor,
          model,
          in_G,
          out_G):
    """Perform one training iteration for the model

    Args:
        input_tensor: (torch.tensor) Tensor representation (1-hot) of sentence
        in input language
        target_tensor: (torch.tensor) Tensor representation (1-hot) of target
        sentence in output language
        model_to_train: (nn.Module: Seq2SeqModel) seq2seq model being trained
        in_G: input group
        out_G: output_group
        enc_optimizer: (torch.optimizer) Optimizer object for model encoder
        dec_optimizer: (torch.optimizer) Optimizer object for model decoder
        loss_fn: (torch.nn.Loss) Loss object used for training
        teacher_forcing_ratio: (float) Ratio with which true word is used as
        input to decoder
    Returns:
        (torch.scalar) Value of loss achieved by model at current iteration
    """
    model.eval()

    transformed_input_tensor = g_transform_data(input_tensor, in_G, device)
    # generate transformed target_tensors only for teacher forcing
    transformed_target_tensor = g_transform_data(target_tensor, out_G, device)
    # example model_output
    # -2.0257, -1.9021, -1.9692, -2.1580, -2.1741, -1.9069, -2.0696, -2.6039],
    # [-2.1432, -2.2621, -2.0530, -1.9556, -2.0481, -2.0793, -2.0876, -2.0397],
    # [-2.0652, -2.0721, -2.0830, -2.0721, -2.0667, -2.0631, -2.1637, -2.0541],
    model_output = model(input_tensor=input_tensor,
                                  target_tensor=target_tensor)
    # example transformed_model_output
    # [-2.0257, -1.9021, -1.9069, -2.1580, -2.1741, -1.9692, -2.0696, -2.6039],
    # [-2.1432, -2.2621, -2.0793, -1.9556, -2.0481, -2.0530, -2.0876, -2.0397],
    # [-2.0652, -2.0721, -2.0631, -2.0721, -2.0667, -2.0830, -2.1637, -2.0541],
    transformed_model_output = g_transform_prob_data(model_output, out_G)
    # example model_transformed_output
    # [-2.0257, -1.9021, -1.9069, -2.1580, -2.1741, -1.9692, -2.0696, -2.6039],
    # [-2.1432, -2.2621, -2.0793, -1.9556, -2.0481, -2.0530, -2.0876, -2.0397],
    # [-2.0652, -2.0721, -2.0631, -2.0721, -2.0667, -2.0830, -2.1637, -2.0541],
    model_transformed_output = model(input_tensor=transformed_input_tensor[1],
                                  target_tensor=transformed_target_tensor[1])

    assert torch.allclose(transformed_model_output, model_transformed_output)


def main(args):
    # get equivariant words
    if args.equivariance == 'verb':
        in_equivariant_words = ['jump', 'run', ]
        out_equivariant_words = ['JUMP', 'RUN']
    elif args.equivariance == 'direction':
        in_equivariant_words = ['right', 'left']
        out_equivariant_words = ['TURN_RIGHT', 'TURN_LEFT']
    else:
        in_equivariant_words = []
        out_equivariant_words = []

    # Load data
    train_pairs, test_pairs = get_scan_split(split=args.split)
    commands, actions = get_invariant_scan_languages(train_pairs, invariances=[])

    # get commands, actions, indices, group generators
    if args.layer_type in ['RNN', 'GRU', 'LSTM']:
        equivariant_commands, equivariant_actions = get_invariant_scan_languages(train_pairs, invariances=[])
        equivariant_commands_indices, equivariant_actions_indices = [equivariant_commands.word2index[word] for word in
                                                                     in_equivariant_words], [
                                                                        equivariant_actions.word2index[word] for word in
                                                                        out_equivariant_words]
        # input group generator and cyclic groups
        in_g = cyclic_group_generator(vocab_size=equivariant_commands.n_words, eq_indices=equivariant_commands_indices)
        in_G = cyclic_group(g=in_g, group_size=len(equivariant_commands_indices),
                            vocab_size=equivariant_commands.n_words)

        # output group generator and cyclic group
        out_g = cyclic_group_generator(vocab_size=equivariant_actions.n_words, eq_indices=equivariant_actions_indices)
        out_G = cyclic_group(g=out_g, group_size=len(equivariant_actions_indices),
                             vocab_size=equivariant_actions.n_words)
    else:
        raise NotImplementedError("Current implementation only supports non-eq seq2seq pretrained models")

        # Initialize model
    if args.layer_type in ['RNN', 'GRU', 'LSTM']:
        model = BasicSeq2Seq(input_language=equivariant_commands,
                             encoder_hidden_size=args.hidden_size,
                             decoder_hidden_size=args.hidden_size,
                             output_language=equivariant_actions,
                             layer_type=args.layer_type,
                             use_attention=args.use_attention,
                             drop_rate=args.drop_rate,
                             bidirectional=args.bidirectional,
                             num_layers=args.num_layers)
    else:
        raise NotImplementedError("Current implementation only supports non-eq seq2seq models")
        # Move model to device and load weights
    model.to(device)

    # initialize equitune model
    equitune_model = EquiSCANModel(pre_model=model, in_G=in_G, out_G=out_G, vocab_size=equivariant_actions.n_words,
                                   eq_word_indices=equivariant_actions_indices,
                                   feature_extracting=args.feature_extracting,
                                   group_type=args.group_type)

    # initialize equitune model
    equizero_model = EquiZeroSCANModel(pre_model=model, in_G=in_G, out_G=out_G, vocab_size=equivariant_actions.n_words,
                                   eq_word_indices=equivariant_actions_indices,
                                   feature_extracting=args.feature_extracting,
                                   group_type=args.group_type)

    # Split off validation set
    val_size = int(len(train_pairs) * args.validation_size)
    random.shuffle(train_pairs)
    train_pairs, val_pairs = train_pairs[val_size:], train_pairs[:val_size]

    # Convert data to torch tensors
    training_pairs = [tensors_from_pair(random.choice(train_pairs), commands, actions)
                      for i in range(args.n_iters)]
    training_eval = [tensors_from_pair(pair, commands, actions)
                     for pair in train_pairs]
    validation_pairs = [tensors_from_pair(pair, commands, actions)
                        for pair in val_pairs]
    testing_pairs = [tensors_from_pair(pair, commands, actions)
                     for pair in test_pairs]

    for iteration in range(1, args.n_iters + 1):
        # Grab iteration translation triplet (input tensor, syntax tensor, output tensor)
        training_pair = training_pairs[iteration - 1]
        iteration_input, iteration_output = training_pair

        # Compute loss (and take one gradient step)
        equitest(input_tensor=iteration_input,
                         target_tensor=iteration_output,
                         model=equitune_model,
                         in_G=in_G,
                         out_G=out_G)

        # Compute loss (and take one gradient step)
        # todo: make sure this test passes
        equizerotest(input_tensor=iteration_input,
                 target_tensor=iteration_output,
                 model=equizero_model,
                 in_G=in_G,
                 out_G=out_G)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model options
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--layer_type',
                        choices=['LSTM', 'GRU', 'RNN', 'GLSTM', 'GGRU', 'GRNN'],
                        default='RNN',
                        help='Type of rnn layers to be used for recurrent components')
    parser.add_argument('--hidden_size',
                        type=int,
                        default=64,
                        help='Number of hidden units in encoder / decoder')
    parser.add_argument('--semantic_size',
                        type=int,
                        default=64,
                        help='Dimensionality of semantic embedding')
    parser.add_argument('--num_layers',
                        type=int,
                        default=1,
                        help='Number of hidden layers in encoder')
    parser.add_argument('--use_attention',
                        dest='use_attention',
                        default=True,
                        action='store_true',
                        help="Boolean to use attention in the decoder")
    parser.add_argument('--bidirectional',
                        dest='bidirectional',
                        default=True,
                        action='store_true',
                        help="Boolean to use bidirectional encoder")
    parser.add_argument('--drop_rate',
                        type=float,
                        default=0.1,
                        help="Dropout drop rate (not keep rate)")

    # Optimization and training hyper-parameters
    parser.add_argument('--split',
                        default='around_right',
                        choices=[None, 'simple', 'add_jump', 'around_right'],
                        help='Each possible split defines a different experiment as proposed by [1]')
    parser.add_argument('--validation_size',
                        type=float,
                        default=0.2,
                        help='Validation proportion to use for early-stopping')
    parser.add_argument('--n_iters',
                        type=int,
                        default=1,
                        help='number of training iterations')
    parser.add_argument('--learning_rate',
                        type=float,
                        default=1.25 * 1e-5,  # set to 1e-5 for around_right
                        help='init learning rate')
    parser.add_argument('--teacher_forcing_ratio',
                        type=float,
                        default=0.5)
    parser.add_argument('--save_dir',
                        type=str,
                        default='./equituned_models/',
                        help='Top-level directory for saving equituned experiment')
    parser.add_argument('--print_freq',
                        type=int,
                        default=1000,
                        help='Frequency with which to print training loss')
    parser.add_argument('--save_freq',
                        type=int,
                        default=1000,
                        help='Frequency with which to save models during training')

    # Equituning options
    parser.add_argument('--pretrained_model_dir',
                        type=str,
                        default='./models/around_right/rnn_RNN_hidden_64_directions_2/seed_0/model0/',
                        help='path to pretrained model')

    parser.add_argument('--group_type',
                        choices=['cyclic', 'none', 'permutation'],
                        default='permutation',
                        help='group type for equivariance')

    parser.add_argument('--equivariance',
                        default='direction',
                        choices=['verb', 'direction', 'none'])

    parser.add_argument('--compute_test_accuracy',
                        dest='compute_test_accuracy',
                        default=True,
                        action='store_true',
                        help="Boolean to print test accuracy")
    parser.add_argument('--feature_extracting',
                        dest='feature_extracting',
                        default=False,
                        action='store_true',
                        help="Boolean to freeze pretrained model")

    args = parser.parse_args()

    main(args)