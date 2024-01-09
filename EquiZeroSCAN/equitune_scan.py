"""
Equitune code
Input: Takes in a non-equivariant pretrained model, a symmetry group or assuming cyclic group, provide words that form the group, n_iters to fine-tune it for.
Output: Equituned model

Test: Use equituned model, but also use the output method used in equituning.
"""

import random
import argparse
import os

import torch
import torch.nn as nn

import perm_equivariant_seq2seq.utils as utils
from utils import set_seed, count_parameters
from perm_equivariant_seq2seq.models import BasicSeq2Seq
from perm_equivariant_seq2seq.equitune_models import EquiSCANModel
from perm_equivariant_seq2seq.data_utils import get_scan_split
from perm_equivariant_seq2seq.utils import tensors_from_pair
from perm_equivariant_seq2seq.data_utils import get_invariant_scan_languages
from perm_equivariant_seq2seq.g_utils import cyclic_group_generator, cyclic_group, g_transform_data, g_inv_transform_prob_data

"""
[1]: Lake and Baroni 2019: Generalization without systematicity: On the 
compositional skills of seq2seq networks
[2]: Bahdanau et al. 2014: Neural machine translation by jointly learning to 
align and translate
[3]: Russin et ak. 2019: Compositional generalization in a deep seq2seq model 
by saparating syntax and semantics
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOS_token = 0
EOS_token = 1

# Parse command-line arguments
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
                    default='add_jump',
                    choices=[None, 'simple', 'add_jump', 'around_right'],
                    help='Each possible split defines a different experiment as proposed by [1]')
parser.add_argument('--validation_size',
                    type=float,
                    default=0.2,
                    help='Validation proportion to use for early-stopping')
parser.add_argument('--n_iters',
                    type=int,
                    default=10000,
                    help='number of training iterations')
parser.add_argument('--learning_rate',
                    type=float,
                    default=5*1e-5,  # set to 2*1e-5 for around_right, 5*1e-5 for add_jump
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
                    default=100,
                    help='Frequency with which to print training loss')
parser.add_argument('--save_freq',
                    type=int,
                    default=100,
                    help='Frequency with which to save models during training')

# Equituning options
parser.add_argument('--pretrained_model_dir',
                    type=str,
                    default='./models/add_jump/rnn_RNN_hidden_64_directions_2/seed_0/model0/',
                    help='path to pretrained model')

parser.add_argument('--group_type',
                    choices=['cyclic', 'permutation', 'none'],
                    default='permutation',
                    help='group type for equivariance')

parser.add_argument('--equivariance',
                    default='verb',
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
parser.add_argument('--count_parameters',
                    dest='count_parameters',
                    default=False,
                    action='store_true',
                    help="Boolean to count parameters in a model")

args = parser.parse_args()

set_seed(args.seed)
args.save_path = os.path.join(args.save_dir,
                              '%s' % args.split,
                              'rnn_%s_hidden_%s_directions_%s' % (
                                  args.layer_type,
                                  args.hidden_size,
                                  2 if args.bidirectional else 1
                              ),
                              'seed_%s' % args.seed)
# Create model directory
if not os.path.isdir(args.save_path):
    os.makedirs(args.save_path)


def equitune(input_tensor,
          target_tensor,
          model_to_train,
          in_G,
          out_G,
          enc_optimizer,
          dec_optimizer,
          loss_fn,
          teacher_forcing_ratio):
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
    model_to_train.train()
    # Forget gradients via optimizers
    enc_optimizer.zero_grad()
    dec_optimizer.zero_grad()

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    model_output = model_to_train(input_tensor=input_tensor,
                                  target_tensor=target_tensor,
                                  use_teacher_forcing=use_teacher_forcing)

    train_loss = 0

    target_length = target_tensor.size(0)
    for di in range(target_length):
        decoder_output = model_output[di]
        train_loss += loss_fn(decoder_output[None, :], target_tensor[di])
        _, decoder_output_symbol = decoder_output.topk(1)
        if decoder_output_symbol.item() == EOS_token:
            break
    train_loss.backward()

    # Clip gradients by norm (5.) and take optimization step
    torch.nn.utils.clip_grad_norm_(model_to_train.pre_model.encoder.parameters(), 5.)
    torch.nn.utils.clip_grad_norm_(model_to_train.pre_model.decoder.parameters(), 5.)
    enc_optimizer.step()
    dec_optimizer.step()

    return train_loss.item() / target_length


def test_accuracy(model_to_test, pairs, in_G, out_G,):
    """Test a model (metric: accuracy) on all pairs in _pairs_

    Args:
        model_to_test: (seq2seq) Model object to be tested
        pairs: (list::pairs) List of list of input/output language pairs
        in_G: input group
        out_G: output group
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
        model_sentence = sentence_ints[:eos_location + 1]
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
            transformed_input_tensor = g_transform_data(input_tensor, in_G, device)

            # get transformed outputs
            transformed_outputs = []
            for i in range(len(transformed_input_tensor)):
                curr_input = transformed_input_tensor[i]
                model_output = model_to_test(input_tensor=curr_input)
                transformed_outputs.append(model_output)

            # add inv_transforms here
            transformed_outputs = torch.stack(transformed_outputs)
            outputs = g_inv_transform_prob_data(transformed_outputs, G=out_G)
            model_output = torch.mean(outputs, dim=0, keepdim=False)
            accuracies.append(sentence_correct(output_tensor, model_output))
    return torch.stack(accuracies).type(torch.float).mean()


if __name__ == '__main__':
    # get equivariant words
    if args.equivariance == 'verb':
        in_equivariant_words = ['jump', 'run',]
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
        if args.count_parameters:
            print(f"num of parameters: {count_parameters(model)}")
    else:
        raise NotImplementedError("Current implementation only supports non-eq seq2seq models")

    # Move model to device and load weights
    pretrained_model_path = os.path.join(args.pretrained_model_dir, "best_validation.pt")
    model_state_dicts = torch.load(pretrained_model_path, map_location=torch.device(device))
    model.to(device)
    model.load_state_dict(model_state_dicts)

    # initialize equitune model
    equitune_model = EquiSCANModel(pre_model=model, in_G=in_G, out_G=out_G, vocab_size=equivariant_actions.n_words,
                                   eq_word_indices=equivariant_actions_indices,
                                   feature_extracting=args.feature_extracting,
                                   group_type=args.group_type)

    # Initialize optimizers
    encoder_optimizer = torch.optim.Adam(model.encoder.parameters(),
                                         lr=args.learning_rate,
                                         weight_decay=0.0001)
    decoder_optimizer = torch.optim.Adam(model.decoder.parameters(),
                                         lr=args.learning_rate,
                                         weight_decay=0.0001)

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

    # Initialize criterion
    criterion = nn.NLLLoss().to(device)

    # Initialize printing / plotting variables
    plot_losses = []
    print_loss_total = 0
    plot_loss_total = 0

    # Enter training loop
    best_acc = 0.
    model_path = utils.create_exp_dir(args)

    val_accuracies = []
    test_accuracies = []
    # If n_iters == 0, then provide the results for zero-shot learning
    if args.n_iters == 0:
        print(f"Zero-shot performance...")
        if args.validation_size > 0.:
            val_acc = test_accuracy(model, validation_pairs, in_G, out_G).item()
            val_accuracies.append(val_acc)
            print('Zero shot validation accuracy %s' % (val_acc))

        if args.compute_test_accuracy:
            test_acc = test_accuracy(model, testing_pairs, in_G, out_G)
            test_accuracies.append(test_acc)
            print("Zero shot model test accuracy: %s" % test_acc.item())

    for iteration in range(0, args.n_iters):
        if (iteration) % 100 == 0:
            print(f"iteration: {iteration}")

        # Grab iteration translation triplet (input tensor, syntax tensor, output tensor)
        training_pair = training_pairs[iteration]
        iteration_input, iteration_output = training_pair

        # Compute loss (and take one gradient step)
        loss = equitune(input_tensor=iteration_input,
                     target_tensor=iteration_output,
                     model_to_train=equitune_model,
                     in_G=in_G,
                     out_G=out_G,
                     enc_optimizer=encoder_optimizer,
                     dec_optimizer=decoder_optimizer,
                     loss_fn=criterion,
                     teacher_forcing_ratio=args.teacher_forcing_ratio)

        print_loss_total += loss
        plot_loss_total += loss

        # Print, plot, etc'
        if (iteration) % (args.print_freq) == 0:
            print_loss_avg = print_loss_total / args.print_freq
            print_loss_total = 0
            print('%s iterations: %s' % (iteration, print_loss_avg))

        if (iteration) % (args.save_freq) == 0:
            # save model if is better
            if args.validation_size > 0.:
                val_acc = test_accuracy(model, validation_pairs, in_G, out_G).item()
                val_accuracies.append(val_acc)
                if val_acc > best_acc:
                    best_acc = val_acc
                    save_path = os.path.join(model_path, 'best_validation.pt')
                    print('Best validation accuracy at iteration %s: %s' % (iteration, val_acc))
                    torch.save(model.state_dict(), save_path)

            if args.compute_test_accuracy:
                test_acc = test_accuracy(model, testing_pairs, in_G, out_G)
                test_accuracies.append(test_acc)
                print("Model test accuracy: %s" % test_acc.item())

    # print loss summary in the end
    print(f"Performance summary...")
    print_loss_avg = print_loss_total / args.print_freq
    print_loss_total = 0
    print('%s iterations: %s' % (args.n_iters, print_loss_avg))

    if args.validation_size > 0.:
        val_acc = test_accuracy(model, validation_pairs, in_G, out_G).item()
        val_accuracies.append(val_acc)
        if val_acc > best_acc:
            best_acc = val_acc
            print('Best validation accuracy at iteration %s: %s' % (args.n_iters, val_acc))
        else:
            print('Best validation accuracy at iteration %s: %s' % (args.n_iters, best_acc))

    if args.compute_test_accuracy:
        test_acc = test_accuracy(model, testing_pairs, in_G, out_G)
        test_accuracies.append(test_acc)
        print("Model test accuracy: %s" % test_acc.item())

    # save logs for test accuracies and val accuracies
    log_dir = "EquiTune_logs"
    import os
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    val_acc_path = os.path.join(log_dir, "val_" + "n_iters" + str(args.n_iters) + args.layer_type + args.split  + "seed" + str(args.seed) + ".pt")
    test_acc_path = os.path.join(log_dir, "test_" + "n_iters" + str(args.n_iters) + args.layer_type + args.split + "seed" + str(args.seed) + ".pt")

    torch.save(torch.tensor(val_accuracies), val_acc_path)
    torch.save(torch.tensor(test_accuracies), test_acc_path)

    # Save fully trained model
    save_path = os.path.join(model_path, 'model_fully_trained.pt')
    torch.save(model.state_dict(), save_path)
