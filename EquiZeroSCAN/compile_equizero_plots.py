# Todo: plot for EquiZero 3K plots over all layer types

import argparse
import torch
import os

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def compile_results(args):
    num_iterations = torch.tensor([i * 200 for i in range(17)])
    color_dict = {"RNN": 'g',
                  "GRU": 'orange',
                  "LSTM": 'b'}
    line_styles_dict = {"EquiZero": "solid",
                        "EquiTune": "dotted"}
    fig = plt.figure()
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    for split in ["add_jump"]:
        for method in ["EquiZero", "EquiTune"]:
            dir_name = method + "_logs"
            for layer_type in ["LSTM", "GRU", "RNN"]:
                results = []
                for seed in range(3):
                    # Note: the 3k results did not have zeroshot results while compiling, hence, we are adding the zeroshot results manually
                    # Note: this is to avoid rerunning the experiments and updated experimental codes have taken care of the issue
                    finetune_filename, zeroshot_filename = args.result_type + "_" + "n_iters" + str(args.n_iters) + layer_type + split + "seed"\
                               + str(seed) + ".pt", args.result_type + "_" + "n_iters" + "0" + layer_type + split + "seed"\
                               + str(seed) + ".pt"
                    finetune_file_path, zeroshot_file_path = os.path.join(dir_name, finetune_filename), os.path.join(dir_name, zeroshot_filename)
                    finetune_result, zeroshot_result = torch.load(finetune_file_path), torch.load(zeroshot_file_path)
                    result = torch.cat((zeroshot_result, finetune_result))
                    results.append(result)

                results_tensor = torch.stack(results)
                results_mean, results_std = torch.mean(results_tensor, dim=0), torch.std(results_tensor, dim=0)

                plt.plot(num_iterations.tolist(), results_mean.tolist(), label=method + layer_type, color=color_dict[layer_type], linestyle=line_styles_dict[method])
                plt.fill_between(num_iterations, results_mean + results_std, results_mean - results_std, alpha=0.05, color=color_dict[layer_type])
    plt.legend()
    plt.title("Equizero vs. equitune for few-shot learning", fontsize=15)
    plt.tight_layout()
    plt.show()
    fig.savefig("Equitune_vs_Equizero.png", dpi=150)
    return


if __name__ == "__main__":
    # e.g. command line prompt
    # python compile_equizero_results.py --method EquiZero --split add_jump --layer_type LSTM --n_iters 10000 --result_type "val"
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', default="EquiZero", type=str, choices=["EquiZero", "EquiTune"])
    parser.add_argument('--split', default="add_jump", type=str, choices=["add_jump", "around_right"])
    parser.add_argument('--layer_type', default="LSTM", type=str, choices=["RNN", "GRU", "LSTM"])
    parser.add_argument('--n_iters', default=3000, type=int)
    parser.add_argument('--result_type', default="test", type=str, choices=["val", "test"])
    args = parser.parse_args()

    compile_results(args)