import logging

import torch
from transformers import AutoModel, AutoTokenizer

from model.modules import MentionEncoder, MemoryModule

from utils import util
from utils import cluster
from utils.cluster import Cluster, ClusterList


class NTMReader(torch.nn.Module):
    def __init__(self, config):
        super(NTMReader, self).__init__()

        # Trainable modules
        self.backbone = AutoModel.from_pretrained(config["encoder_name"])
        self.tokenizer = AutoTokenizer.from_pretrained(config["encoder_name"])
        self.mention_encoder = MentionEncoder(config)
        self.ntm = MemoryModule(config)

        # Training parameters
        self.max_span_width = config["max_span_width"]
        self.memory_limit = config["memory_limit"]
        self.negative_sample_rate = config["negative_sample_rate"]

        # Inference parameters
        self.threshold = config["threshold"]
        self.mention_classifier = config["mentions"]
        self.singleton = config["singleton_eval"]

        # Debugging
        self.correct_new = 0
        self.correct_attach = 0
        self.wrong_new = 0
        self.wrong_attach = 0
        self.loss_count = 0
        self.sampled_loss_count = 0

        # Move to GPU
        self.device = config["device"]
        self.to(self.device)

        config["hidden_size"] = self.backbone.config.hidden_size

        # Logging
        logging.info("Loaded pretrained model {}".format(config["encoder_name"]))
        logging.info(
            "Number of backbone parameters: {}".format(
                util.count_parameters(self.backbone)
            )
        )
        logging.info(
            "Number of other trainable parameters: {}".format(
                util.count_parameters(self) - util.count_parameters(self.backbone)
            )
        )

    def reset_metrics(self):
        self.correct_new = 0
        self.correct_attach = 0
        self.wrong_new = 0
        self.wrong_attach = 0

    def compute_attach_stats(self, best_cluster_idx, gold_cluster_id):
        if best_cluster_idx != 0:
            if best_cluster_idx == gold_cluster_id:
                self.correct_attach += 1
            else:
                self.wrong_attach += 1
        else:
            if gold_cluster_id == 0:
                self.correct_new += 1
            else:
                self.wrong_new += 1

    def encode(self, **kwargs):
        outputs = self.backbone.encoder(**kwargs)
        return outputs.last_hidden_state

    def decode(self, encoder_outputs, **kwargs):
        return self.backbone(encoder_outputs=encoder_outputs, **kwargs)

    def prepare_document(self, document):

        # input_data: ['doc_key', 'sentences', 'clusters', 'sentence_map', 'subtoken_map', 'antecedent_map']
        text_segments = document["sentences"]
        tokenized = tokenizer(
            text_segments,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64,
            return_offsets_mapping=True,
        )
        get_c2t_map = lambda offset: {
            x: i for i, (s, e) in enumerate(offset) for x in range(s, e)
        }
        c2t_map = list(map(get_c2t_map, tokenized["offset_mapping"]))

        if "antecedent_map" in document:
            clusters_tokenized = []
            for c in clusters[x]:
                clusters_tokenized = []
                for s, e in c:
                    t_s, t_e = (
                        c2t_map[s],
                        (c2t_map[e - 1] + 1),
                    )  # span end index is exclusive
                    clusters_tokenized[-1].add([t_s, t_e])

        return tokenized, clusters_tokenized

    def mini_batch(self, documents):

        # documents: List[dict]
        # each document: {'sentences': [], 'clusters': [], 'sentence_map': [], 'subtoken_map': [], 'antecedent_map': []}
        pass

    def forward(self, segment, model_data, entities, start_idx, metrics=True):

        import ipdb

        ipdb.set_trace()

        encoder_repr = self.encode(segment)  # [batch_size, seq_len, hidden_size]
        span_embeddings, mention_scores, top_spans = self.mention_encoder(
            encoder_repr, tokenized["attention_mask"]
        )

        # sent_gen = util.get_sentence_iter(sentences, top_spans, start_idx, None, self.cluster)
        # spans_loss = self.resolve_local(clusters, sent_gen, model_data["antecedent_map"], span_embeddings, train=train, metrics=metrics) #  encoder_repr.detach() ??


if __name__ == "__main__":

    from transformers import AutoTokenizer
    import utils.util as util

    config = util.initialize_from_env()

    model = NTMReader(config)

    tokenizer = AutoTokenizer.from_pretrained(config["encoder_name"])
    sentence = "This is a test sentence."
    tokenized = tokenizer.batch_encode_plus([sentence,], return_tensors="pt")
    encoded_repr = model.encode(**tokenized)
    print("Testing encoding:", sentence)
    print("Encoded representation shape:", encoded_repr.shape)
