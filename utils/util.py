from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import errno
import logging
import json
import _jsonnet
import torch
import random


def count_parameters(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def initialize_from_env(eval_test=False, use_overrides=True):
    name = sys.argv[1]
    overrides = {}
    if len(sys.argv) > 2 and use_overrides:
        for item in sys.argv[2:]:
            key, value = item.split("=", 1)
            try:
                overrides[key] = json.loads(value)
            except:
                overrides[key] = value

    config = json.loads(_jsonnet.evaluate_file("jsonnets/experiments.jsonnet"))[name]

    # Put everything in override
    config.update(overrides)

    mkdirs(config["log_dir"])
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(config["log_dir"] + "/out.log"),
            logging.StreamHandler(),
        ],
    )

    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.info("Running experiment: {}".format(name))
    logging.info(json.dumps(config, indent=2))
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    config["device"] = device
    if "load_path" not in config:
        config["load_path"] = config["log_path"]
    return config


def mkdirs(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
    return path


def load_params(module, log_path, key):
    try:
        checkpoint = torch.load(log_path, map_location="cpu")
        logging.info(f"Found checkpoint at {log_path}, loading instead.")
    except:
        logging.info(f"Checkpoint not found at {log_path}")
        return
    try:
        missing, unexpected = module.load_state_dict(checkpoint[key], strict=False)
        if missing or unexpected:
            print(
                f"Did not find (using defaults):{str(missing)}...\n\n"
                + f"Unexpected params (ignoring): {unexpected}"
            )
    except Exception as e1:
        try:
            module.load_state_dict(checkpoint[key])
            logging.info(
                f"Module cannot load with keyword strict=False. Loading with its default."
            )
        except Exception as e2:
            logging.info(f"Unable to load checkpoint for {key}: {e1}, {e2}")
            return


def load_data(path, num_examples=None):
    if path is None or not path:
        return []

    def load_line(line):
        example = json.loads(line)
        # Need to make antecedent dict
        clusters = [sorted(cluster) for cluster in example["clusters"]]
        antecedent_map = {}
        for cluster in clusters:
            antecedent_map[tuple(cluster[0])] = "0"
            for span_idx in range(1, len(cluster)):
                antecedent_map[tuple(cluster[span_idx])] = [
                    tuple(span) for span in cluster[:span_idx]
                ]
        example["antecedent_map"] = antecedent_map
        return example

    # TODO:
    # Add tokenization and mapping clusters here.

    with open(path) as f:
        data = [load_line(l) for l in f.readlines()]
        if num_examples is not None:
            data = data[:num_examples]
        logging.info("Loaded {} examples.".format(len(data)))
        return data


def flatten(l):
    return [item for sublist in l for item in sublist]


def safe_add(tensor1, tensor2):
    # None is the additive identity and the result can be backpropped
    if tensor1 is None:
        return tensor2
    if tensor2 is None:
        return tensor1
    else:
        return tensor1 + tensor2


def get_cuda_memory_allocated():
    GB = 1073741824  # 1024 ** 3
    if torch.cuda.is_available():
        memory = torch.cuda.memory_allocated() / GB
    else:
        memory = 0.0
    return memory


def mention_maps(predicted_clusters, gold_clusters):
    gold_clusters = [tuple(tuple(m) for m in gc) for gc in gold_clusters]
    predicted_clusters = [tuple(tuple(m) for m in pc) for pc in predicted_clusters]
    mention_to_gold = {}
    for gc in gold_clusters:
        for mention in gc:
            mention_to_gold[mention] = gc
    mention_to_predicted = {}
    for pc in predicted_clusters:
        for mention in pc:
            mention_to_predicted[mention] = pc
    return predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold


def make_evict_function(params):
    if type(params) != dict or "name" not in params:
        return lambda cluster, idx: False

    if params["name"] == "singletons":
        return lambda cluster, idx: (
            len(cluster) == 1 and idx - cluster.start > params["distance"]
        )
    elif params["name"] == "trunc_linscale":
        return lambda cluster, idx: (
            len(cluster) == 1
            and idx - cluster.start > params["distance"]
            or idx - cluster.start > 2 * params["distance"]
        )
    else:
        return lambda cluster, idx: False


def get_minibatch_iter(list, chunk_size):
    for i in range(0, len(list), chunk_size):
        yield list[i : i + chunk_size]


def set_seed(config):
    # There is still nondeterminism somewhere
    logging.info(f"Setting seed to {config['seed']}")
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
