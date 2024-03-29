# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import random
import argparse
import os

import numpy as np
import torch

import perm_equivariant_seq2seq.utils as utils
from perm_equivariant_seq2seq.models import BasicSeq2Seq
from perm_equivariant_seq2seq.equivariant_models import EquiSeq2Seq
from perm_equivariant_seq2seq.data_utils import get_scan_split, get_equivariant_scan_languages, get_invariant_scan_languages
from perm_equivariant_seq2seq.utils import tensors_from_pair, tensor_from_sentence
from perm_equivariant_seq2seq.symmetry_groups import get_permutation_equivariance

np.set_printoptions(linewidth=np.inf)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOS_token = 0
EOS_token = 1


parser = argparse.ArgumentParser()
# Experiment options
parser.add_argument('--seed',
                    default=0,
                    type=int,
                    help='seed argument')
parser.add_argument('--experiment_dir',
                    default='models/simple/rnn_RNN_hidden_64_directions_2/seed_0/model0/',
                    type=str, 
                    help='Path to experiment directory, should contain args and model')
parser.add_argument('--validation_size',
                    type=float,
                    default=0.2,
                    help='Validation proportion to use for early-stopping')
parser.add_argument('--best_validation', 
                    dest='best_validation', 
                    default=True,
                    action='store_true',
                    help="Boolean indicating whether to use best validation model")
parser.add_argument('--fully_trained', 
                    dest='fully_trained', 
                    default=False, 
                    action='store_true',
                    help="Boolean indicating whether to use fully trained model")
parser.add_argument('--iterations', 
                    type=int, 
                    default=0, 
                    help='If not fully trained, how many training iterations.')
parser.add_argument('--compute_train_accuracy', 
                    dest='compute_train_accuracy', 
                    default=False, 
                    action='store_true',
                    help="Boolean to evaluate train accuracy")
parser.add_argument('--compute_test_accuracy', 
                    dest='compute_test_accuracy', 
                    default=False, 
                    action='store_true',
                    help="Boolean to evaluate test accuracy")
parser.add_argument('--print_translations', 
                    type=int, 
                    default=0,
                    help="Print a small number of translations from the test set")
parser.add_argument('--print_param_nums', 
                    dest='print_param_nums', 
                    default=False, 
                    action='store_true',
                    help="Print the number of model parameters")
# Model options
parser.add_argument('--hidden_size', 
                    type=int, 
                    default=64, 
                    help='Number of hidden units in encoder / decoder')
parser.add_argument('--num_layers',
                    type=int,
                    default=1,
                    help='num of layers used for RNN.')
parser.add_argument('--drop_rate',
                    type=float,
                    default=0.1,
                    help='drop rate for RNN.')
parser.add_argument('--layer_type', 
                    choices=['GGRU', 'GRNN', 'GLSTM', 'RNN', 'GRU', 'LSTM'],
                    default='RNN',
                    help='Type of rnn layers to be used for recurrent components')
parser.add_argument('--use_attention', 
                    dest='use_attention', 
                    default=True,
                    action='store_true',
                    help="Boolean to use attention in the decoder")
parser.add_argument('--bidirectional', 
                    dest='bidirectional', 
                    default=True,
                    action='store_true',
                    help="Boolean to use bidirectional encoder.")
# Equivariance options:
parser.add_argument('--equivariance',
                    default='none',
                    choices=['verb', 'direction', 'verb+direction', 'none'])
# Optimization and training hyper-parameters
parser.add_argument('--split',
                    default='around_right',
                    help='Each possible split defines a different experiment as proposed by [1]',
                    choices=[None, 'simple', 'add_jump',
                             'length_generalization', 'around_right', 
                             'opposite_right'])
parser.add_argument('--weight_decay', 
                    type=float, 
                    default=0., 
                    help='Weight decay for optimizer')
parser.add_argument('--batch_size', 
                    type=int, 
                    default=1, 
                    help='batch size for training (unused)')
parser.add_argument('--validation_size', 
                    type=float, 
                    default=0.2, 
                    help='Validation proportion to use for early-stopping')
parser.add_argument('--n_iters', 
                    type=int, 
                    default=200000, 
                    help='number of training iterations')
parser.add_argument('--learning_rate', 
                    type=float, 
                    default=1e-4, 
                    help='init learning rate')
parser.add_argument('--teacher_forcing_ratio', 
                    type=float, 
                    default=0.5)
parser.add_argument('--save_dir', 
                    type=str, 
                    default='./models/', 
                    help='Top-level directory for saving experiment')
parser.add_argument('--print_freq', 
                    type=int, 
                    default=1000, 
                    help='Frequency with which to print training loss')
parser.add_argument('--save_freq', 
                    type=int, 
                    default=200000, 
                    help='Frequency with which to save models during training')

args = parser.parse_args()

def evaluate(model_to_eval,
             inp_lang,
             out_lang,
             sentence):
    """Decode one sentence from input -> output language with the model

    Args:
        model_to_eval: (nn.Module: Seq2SeqModel) seq2seq model being evaluated
        inp_lang: (Lang) Language object for input language
        out_lang: (Lang) Language object for output language
        sentence: (torch.tensor) Tensor representation (1-hot) of sentence in 
        input language
    Returns:
        (list) Words in output language as decoded by model
    """
    model.eval()
    with torch.no_grad():
        input_sentence = tensor_from_sentence(inp_lang, sentence)
        model_output = model_to_eval(input_tensor=input_sentence)

        decoded_words = []
        for di in range(model_to_eval.max_length):
            topv, topi = model_output[di].data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(out_lang.index2word[topi.item()])
        return decoded_words


def test_accuracy(model_to_test, pairs):
    """Test a model (metric: accuracy) on all pairs in _pairs_

    Args:
        model_to_test: (seq2seq) Model object to be tested
        pairs: (list::pairs) List of list of input/output language pairs
    Returns:
        (float) accuracy on test pairs
    """
    def sentence_correct(target, model_sentence):
        # First, extract sentence up to EOS
        _, sentence_ints = model_sentence.data.topk(1)
        # If there is no EOS token, take the complete list
        try:
            eos_location = (sentence_ints == EOS_token).nonzero()[0][0]
        except:
            eos_location = len(sentence_ints) - 2
        model_sentence = sentence_ints[:eos_location+1]
        # Check length is correct
        if len(model_sentence) != len(target):
            return torch.tensor(0, device=device)
        else:
            correct = model_sentence == target
            return torch.prod(correct).to(device)

    accuracies = []
    model.eval()
    with torch.no_grad():
        for pair in pairs:
            input_tensor, output_tensor = pair
            model_output = model_to_test(input_tensor=input_tensor)
            accuracies.append(sentence_correct(output_tensor, model_output))
    return torch.stack(accuracies).type(torch.float).mean()


if __name__ == '__main__':
    # Make sure all data is contained in the directory and load arguments
    args_path = os.path.join(args.experiment_dir, "commandline_args.txt")
    if args.best_validation:
        model_path = os.path.join(args.experiment_dir, "best_validation.pt")
    elif args.fully_trained:
        model_path = os.path.join(args.experiment_dir, 
                                  "model_fully_trained.pt")
    else:
        model_path = os.path.join(args.experiment_dir, 
                                  "model_iteration%s.pt" % args.iterations)
    assert os.path.exists(args.experiment_dir), \
        "Experiment directory not found"
    assert os.path.exists(model_path), "Model number not found in directory"
    assert os.path.exists(args_path), \
        "Argparser details directory not found in directory"
    experiment_arguments = utils.load_args_from_txt(parser, args_path)

    # Load data
    experiment_arguments.split = args.split  # This helps test models on splits than the ones they were trained on
    train_pairs, test_pairs = get_scan_split(split=experiment_arguments.split)

    if experiment_arguments.equivariance == 'verb':
        in_equivariances = ['jump', 'run', 'walk', 'look']
        out_equivariances = ['JUMP', 'RUN', 'WALK', 'LOOK']
    elif experiment_arguments.equivariance == 'direction':
        in_equivariances = ['right', 'left']
        out_equivariances = ['TURN_RIGHT', 'TURN_LEFT']
    elif experiment_arguments.equivariance == 'verb+direction':
        in_equivariances = ['jump', 'run', 'walk', 'look', 'right', 'left']
        out_equivariances = ['JUMP', 'RUN', 'WALK', 'LOOK', 'TURN_RIGHT', 'TURN_LEFT']
    else:
        in_equivariances, out_equivariances = [], []

    # get commands and actions
    if args.layer_type in ['GRNN', 'GGRU', 'GLSTM']:
        equivariant_commands, equivariant_actions = \
            get_equivariant_scan_languages(pairs=train_pairs,
                                           input_equivariances=in_equivariances,
                                           output_equivariances=out_equivariances)
    else:
        experiment_arguments.equivariance = 'none'  # needs to be explicitly set for obtaining the symmetry groups next
        equivariant_commands, equivariant_actions = get_invariant_scan_languages(train_pairs, invariances=[])

    # get symmetry groups
    if experiment_arguments.equivariance == 'verb+direction':
        from perm_equivariant_seq2seq.symmetry_groups import VerbDirectionSCAN

        input_symmetry_group = VerbDirectionSCAN(
            num_letters=equivariant_commands.n_words,
            first_equivariant=equivariant_commands.num_fixed_words + 1
        )
        output_symmetry_group = VerbDirectionSCAN(
            num_letters=equivariant_actions.n_words,
            first_equivariant=equivariant_actions.num_fixed_words + 1
        )
    elif experiment_arguments.equivariance == 'none':
        pass
    else:
        input_symmetry_group = get_permutation_equivariance(equivariant_commands)
        output_symmetry_group = get_permutation_equivariance(equivariant_actions)

    # Initialize model
    if args.layer_type in ['GRNN', 'GGRU', 'GLSTM']:
        model = EquiSeq2Seq(input_symmetry_group=input_symmetry_group,
                            output_symmetry_group=output_symmetry_group,
                            input_language=equivariant_commands,
                            encoder_hidden_size=experiment_arguments.hidden_size,
                            decoder_hidden_size=experiment_arguments.hidden_size,
                            output_language=equivariant_actions,
                            layer_type=args.layer_type,
                            use_attention=experiment_arguments.use_attention,
                            bidirectional=experiment_arguments.bidirectional)
    else:
        model = BasicSeq2Seq(input_language=equivariant_commands,
                             encoder_hidden_size=args.hidden_size,
                             decoder_hidden_size=args.hidden_size,
                             output_language=equivariant_actions,
                             layer_type=args.layer_type,
                             use_attention=args.use_attention,
                             drop_rate=args.drop_rate,
                             bidirectional=args.bidirectional,
                             num_layers=args.num_layers)

    # Move model to device and load weights
    model_state_dicts = torch.load(model_path, map_location=torch.device(device))
    model.to(device)
    model.load_state_dict(model_state_dicts)

    # Split off validation set
    val_size = int(len(train_pairs) * args.validation_size)
    random.shuffle(train_pairs)
    train_pairs, val_pairs = train_pairs[val_size:], train_pairs[:val_size]

    # Convert data to torch tensors
    training_pairs = [tensors_from_pair(random.choice(train_pairs), equivariant_commands, equivariant_actions)
                      for i in range(args.n_iters)]
    training_eval = [tensors_from_pair(pair, equivariant_commands, equivariant_actions)
                     for pair in train_pairs]
    validation_pairs = [tensors_from_pair(pair, equivariant_commands, equivariant_actions)
                        for pair in val_pairs]
    testing_pairs = [tensors_from_pair(pair, equivariant_commands, equivariant_actions)
                     for pair in test_pairs]

    val_accuracies = []
    test_accuracies = []
    if args.validation_size > 0.:
        val_acc = test_accuracy(model, validation_pairs).item()
        val_accuracies.append(val_acc)
        print('Validation accuracy %s' % (val_acc))

    if args.compute_test_accuracy:
        test_acc = test_accuracy(model, testing_pairs)
        test_accuracies.append(test_acc)
        print("Test accuracy: %s" % test_acc.item())

    # save logs for test accuracies and val accuracies
    # This test works for zeroshot testing for both equivariant and non-equivariant models
    if args.equivariance == 'none':
        log_dir = "Non_Equivariant_logs"
        import os
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        val_acc_path = os.path.join(log_dir, "val_" + "n_iters" + str(args.n_iters) + args.layer_type + args.split  + "seed" + str(args.seed) + ".pt")
        test_acc_path = os.path.join(log_dir, "test_" + "n_iters" + str(args.n_iters) + args.layer_type + args.split + "seed" + str(args.seed) + ".pt")
    else:
        log_dir = "Group_Equivariant_logs"
        import os

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        val_acc_path = os.path.join(log_dir, "val_" + "n_iters" + str(
            args.n_iters) + args.layer_type + args.split + "seed" + str(args.seed) + ".pt")
        test_acc_path = os.path.join(log_dir, "test_" + "n_iters" + str(
            args.n_iters) + args.layer_type + args.split + "seed" + str(args.seed) + ".pt")

    torch.save(torch.tensor(val_accuracies), val_acc_path)
    torch.save(torch.tensor(test_accuracies), test_acc_path)

    # Todo: remove the following if unnecessary
    # # Compute accuracy and print some translation
    # if args.compute_train_accuracy:
    #     train_acc = test_accuracy(model, training_eval)
    #     print("Model train accuracy: %s" % train_acc.item())
    # if args.compute_test_accuracy:
    #     test_acc = test_accuracy(model, testing_pairs)
    #     print("Model test accuracy: %s" % test_acc.item())
    # if args.print_param_nums:
    #     print("Model contains %s params" % model.num_params)
    # for i in range(args.print_translations):
    #     pair = random.choice(test_pairs)
    #     print('>', pair[0])
    #     print('=', pair[1])
    #     output_words = evaluate(model, equivariant_commands,
    #                             equivariant_actions, pair[0])
    #     output_sentence = ' '.join(output_words)
    #     print('<', output_sentence)
    #     print('')
