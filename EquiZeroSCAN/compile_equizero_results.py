import argparse
import torch
import os


def compile_results(args):
    dir_name = args.method + "_logs"

    results = []
    for seed in range(3):
        filename = args.result_type + "_" + "n_iters" + str(args.n_iters) + args.layer_type + args.split + "seed"\
                   + str(seed) + ".pt"
        file_path = os.path.join(dir_name, filename)
        result = torch.load(file_path)
        results.append(result)

    results_tensor = torch.stack(results)
    mean, std = torch.mean(results_tensor), torch.std(results_tensor)

    return mean, std


if __name__ == "__main__":
    # e.g. command line prompt
    # python compile_equizero_results.py --method EquiZero --split add_jump --layer_type LSTM --n_iters 10000 --result_type "val"
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', default="EquiZero", type=str, choices=["EquiZero", "EquiTune"])
    parser.add_argument('--split', default="add_jump", type=str, choices=["add_jump", "around_right"])
    parser.add_argument('--layer_type', default="RNN", type=str, choices=["RNN", "GRU", "LSTM"])
    parser.add_argument('--n_iters', default=0, type=int)
    parser.add_argument('--result_type', default="test", type=str, choices=["val", "test"])
    args = parser.parse_args()

    # each run must return
    #   - mean and std for each case in the table
    mean, std = compile_results(args)
    print(f"mean: {mean};  std: {std}")