import clip
import numpy as np


def load_model(args):
    model, preprocess = clip.load(args.model_name)
    model.to(args.device)
    if args.verbose:
        input_resolution = model.visual.input_resolution
        context_length = model.context_length
        vocab_size = model.vocab_size

        print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
        print("Input resolution:", input_resolution)
        print("Context length:", context_length)
        print("Vocab size:", vocab_size)

    return model, preprocess