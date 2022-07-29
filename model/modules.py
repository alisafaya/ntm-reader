import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from allennlp.data.dataset_readers.dataset_utils.span_utils import enumerate_spans
from allennlp.modules.span_extractors.endpoint_span_extractor import EndpointSpanExtractor
from allennlp.nn import util as nn_util

class MentionEncoder(nn.Module):
	def __init__(self, config):
		super(MentionEncoder, self).__init__()

		self.hidden_size = config["hidden_size"]
		self.max_span_width = config["max_span_width"]
		self.top_span_ratio = config["top_span_ratio"]

		self.span_encoder = EndpointSpanExtractor(
			input_dim=self.hidden_size,
			combination="x,y,x*y",
			num_width_embeddings=self.max_span_width,
			span_width_embedding_dim=config["span_width_features"]
		)

		self.span_encoder_dim = self.span_encoder.get_output_dim()
		self.span_encoder_mlp = nn.Sequential(
			nn.Dropout(config["dropout"]),
			nn.Linear(self.span_encoder_dim, self.hidden_size),
			nn.GELU()
		)

		self.mention_scorer = nn.Linear(self.hidden_size, 1)
		self.device = config["device"]

	def forward(self, segment_embeddings, segment_mask, gold_spans=None):

		# segment_embeddings: [batch_size, num_words, hidden_size]
		# segment_mask: [batch_size, num_words]

		# This is the output of the pre-trained encoder.
		batch_size, num_words, hidden_size = segment_embeddings.shape

		if gold_spans is not None:
			candidate_spans = torch.LongTensor(gold_spans).to(segment_embeddings.device)
		else:
			candidate_spans = enumerate_spans(torch.arange(num_words), offset=0, max_span_width=self.max_span_width) # [num_candidates, 2]
			candidate_spans = sorted(candidate_spans[::-1], key=lambda x: x[0]) # spans that end last come last.
			candidate_spans = torch.LongTensor([candidate_spans] * batch_size).to(segment_embeddings.device)

		# encode all possible spans. 
		span_embeddings = self.span_encoder(segment_embeddings, candidate_spans, segment_mask) # [batch_size, max_num_spans, span_encoder_dim]

		# project span representations.
		span_embeddings = self.span_encoder_mlp(span_embeddings) # [batch_size, max_num_spans, hidden_size]

		# Get mention scores.
		mention_scores = self.mention_scorer(span_embeddings) # [batch_size, max_num_spans, 1]

		if gold_spans is not None:
			return span_embeddings, mention_scores.squeeze(), candidate_spans
		else:
			# Mask mentions for invalid spans. i.e. spans with paddings.
			i, j = torch.where(candidate_spans[:, :, -1] >= segment_mask.sum(-1).view(-1, 1))
			mask = torch.zeros_like(mention_scores)
			mask[i, j, :] = -1e6
			mention_scores = mention_scores + mask

			k = int(num_words * self.top_span_ratio) // (1 + int(self.training))

			_, top_k_indices = torch.topk(mention_scores.squeeze(), k, sorted=False) # sorted=False to reduce overhead.

			if self.training:
				sampled_indices = torch.randperm(mention_scores.size(1))[:k].view(1, -1).repeat(batch_size, 1).to(top_k_indices.device)
				top_k_indices = torch.cat([ top_k_indices, sampled_indices ], dim=1)

			top_k_indices, _ = torch.sort(top_k_indices, 1) # to keep the sequential order of candidate spans.
			top_scores = nn_util.batched_index_select(mention_scores, top_k_indices).squeeze() # [batch_size, k, hidden_size]
			top_span_embs = nn_util.batched_index_select(span_embeddings, top_k_indices) # [batch_size, k]
			top_spans = nn_util.batched_index_select(candidate_spans, top_k_indices) # [batch_size, k, 2]

			return top_span_embs, top_scores, top_spans


class NTMAdresser(nn.Module):
	def __init__(self, h_dim, dropout=0.25):
		super(NTMAdresser, self).__init__()

		self.d_k = h_dim
		self.scorer = nn.Sequential(
			nn.Linear(h_dim * 2, h_dim), 
			nn.GELU(), 
			nn.Dropout(dropout), 
			nn.Linear(h_dim, 1)
		)

		self.new_address = nn.Parameter(torch.randn(1, 1, h_dim) / math.sqrt(h_dim))
		self.null_address = nn.Parameter(torch.randn(1, 1, h_dim) / math.sqrt(h_dim))

	def forward(self, query, keys, entity_size, mention_score):

		# query: [batch_size, d_k]
		# keys: [batch_size, num_entities, d_k]
		# entity_size: [batch_size]
		# mention_score: [batch_size]

		batch_size, num_entities, _ = keys.shape

		# 0) Concatenate new address and null address vectors to allow for not-a-mention or new entries in memory.
		keys = torch.cat([self.null_address.repeat(batch_size, 1, 1), keys, self.new_address.repeat(batch_size, 1, 1)], dim=1)

		# 1) Get similarity scores
		query = query.unsqueeze(1).repeat(1, num_entities + 2, 1)
		query_keys = torch.cat([query, keys], dim=-1)
		scores = self.scorer(query_keys).squeeze() # [batch_size, 2 + num_entities]

		# 2) Incorporate score from mention classification since we are doing joint scoring.
		# This way we subtracte the mention score from the null score, and add it to the other scores. 
		inverse_mask = torch.ones_like(scores)
		inverse_mask[:, 0] = -1.0 
		scores = scores + (mention_score.view(batch_size, 1) * inverse_mask)

		# 3) Mask out padding entities.
		mask = torch.zeros((batch_size, num_entities + 2), device=scores.device, dtype=scores.dtype)
		for b in range(batch_size):
			mask[b, (entity_size[b] + 1): -1] = -1e6
		scores = scores + mask

		return scores


class MemoryModule(nn.Module):
	def __init__(self, config):
		super(MemoryModule, self).__init__()
		"""
			This module is responsible for resolving coreference and updating entity memory entries similar to the NTM.
		"""
		self.hidden_size = config["hidden_size"]

		# Addresser module to get similarity scores of mentions for each memory entry.
		self.ntm_addresser = NTMAdresser(self.hidden_size, dropout=config["dropout"])
		
		# GRU Cell to manage memory write operations.
		self.gru_cell = nn.GRUCell(self.hidden_size * 3, self.hidden_size)

		# Attention module over entity representations
		self.query_affine = nn.Linear(self.hidden_size, self.hidden_size)
		self.context_attn = nn.MultiheadAttention(self.hidden_size, config["context_attn_heads"], batch_first=True)

		self.device = config["device"]

	def update(self, context, query_span, entity_memory, scores):

		# context: [batch_size, num_words, hidden_size]
		# query_span: [batch_size, hidden_size]
		# entity_memory: [batch_size, num_entities, hidden_size]
		# scores: [batch_size, num_entities + 2]

		batch_size, num_entities, hidden_size = entity_memory.shape

		scores = scores[:, 1:] # [batch_size, num_entities + 1] here we remove the null score.
		query_span = query_span.unsqueeze(1)

		# New information to incorporate from both query_span and the prev_entity_memory using the read head of NTM.
		prev_entity_memory = torch.cat([entity_memory, query_span], dim=1) # [batch_size, num_entities + 1, hidden_size]

		# 1) NTM Read from memory and aggregate information from other entries based on the scores.
		read_out = scores.unsqueeze(2) @ (scores.unsqueeze(1) @ prev_entity_memory) # [batch_size, 1 + num_entities, hidden_size]

		# 2) Attention over the context.
		# The context is the embeddings of the text segment. Here we give them as a context during the memory update step.
		context_attn, _ = self.context_attn(self.query_affine(prev_entity_memory), context, context)

		# 2) Query span embedding.
		query_span = query_span.repeat(1, num_entities + 1, 1) # [batch_size, num_entities + 1, hidden_size]

		focus_memory = torch.cat([ 
				read_out,
				context_attn,
				query_span
			], dim=-1) # [batch_size, num_entities + 1, hidden_size * 3]

		# Reshape to be compatible with GRU cell.
		focus_memory = focus_memory.view((num_entities + 1) * batch_size, -1)
		prev_entity_memory = prev_entity_memory.view((num_entities + 1) * batch_size, -1)

		# update the memory
		entity_memory = self.gru_cell(focus_memory, prev_entity_memory) # [ (batch_size * num_entities + 1) , hidden_size]

		# Reshape back.
		entity_memory = entity_memory.view(batch_size, num_entities + 1, hidden_size) # [batch_size, num_entities + 1, hidden_size]
		return entity_memory

	def forward(self, context, query_span, mention_score, entity_memory=None, entity_size=None, gold_entity_ids=None, use_teacher_forcing=False):

		# context: [batch_size, num_words, hidden_size]
		# query_span: [batch_size, hidden_size]
		# entity_memory: [batch_size, num_entities, hidden_size]
		# entity_size: [batch_size]
		# mention_score: [batch_size]
		# gold_entity_ids: [batch_size]

		# +2 for null and new entries. 
		# Where 0 means null, and (entity.shape[1] + 1) means new entry.

		batch_size, _, hidden_size = context.shape
	
		if entity_memory is None:
			entity_memory = torch.empty(batch_size, 0, hidden_size, device=query_span.device)
			entity_size = torch.zeros(batch_size, device=query_span.device, dtype=torch.long)
			num_entities = 0
		else:
			num_entities = entity_memory.size(1)

		# Get the scores from the addresser.
		predict_scores = self.ntm_addresser(query_span, entity_memory, entity_size, mention_score) # [batch_size, 2 + num_entities]

		# If gold_cluster_id is provided, then calculate the loss.
		if gold_entity_ids is not None:
			gold_entity_ids = gold_entity_ids.clone()
			# edit gold_entity_ids to support minibatching. 
			gold_entity_ids[gold_entity_ids > entity_size] = entity_memory.size(1) + 1
			cluster_loss = F.cross_entropy(predict_scores, gold_entity_ids)
		else:
			cluster_loss = None

		# Teacher forcing on memory addresser overrides the addresser scores to always predict the gold cluster id.
		if use_teacher_forcing and gold_entity_ids is not None:
			gold_scores = torch.zeros_like(predict_scores)
			gold_scores[torch.arange(batch_size), gold_entity_ids] = 1e2
			gold_scores = gold_scores.clamp(min=1e-7)
			_scores = gold_scores
		else:
			_scores = predict_scores

		# Normalize the scores.
		_scores = F.softmax(_scores, dim=-1)

		# Get the best cluster id.
		best_cluster_idx = torch.argmax(_scores, dim=1)

		# Don't update the memory where the best cluster idx == 0.
		to_be_updated, = torch.where(best_cluster_idx != 0)
		not_a_mention, = torch.where(best_cluster_idx == 0)

		new_entity_size = entity_size.clone()
		new_entity_memory = torch.zeros(batch_size, num_entities + 1, self.hidden_size, device=entity_memory.device, dtype=entity_memory.dtype)

		# Update the memory with the new information from the query_span embedding based on the scores.
		if len(to_be_updated) > 0:		
			updated_memory = self.update(context[to_be_updated], query_span[to_be_updated], entity_memory[to_be_updated], _scores[to_be_updated])

			new_entity_memory[to_be_updated] = updated_memory
			new_entity_memory[to_be_updated, entity_size[to_be_updated]] = updated_memory[:, -1] # Since the new entry is always the last one, regardless of the entity size.

			new_entry_exists = (best_cluster_idx[to_be_updated] == new_entity_memory.size(1)) # If a new entry is there increase the entity count.
			new_entity_size[to_be_updated] += new_entry_exists

			best_cluster_idx[best_cluster_idx == new_entity_memory.size(1)] = new_entity_size[best_cluster_idx == new_entity_memory.size(1)]

		# Keep the memory unchanged where the best cluster idx == 0. i.e. not a mention.
		if len(not_a_mention) > 0:
			new_entity_memory[not_a_mention] = torch.cat([ entity_memory[not_a_mention],
															torch.zeros(len(not_a_mention), 1, hidden_size, device=entity_memory.device, dtype=entity_memory.dtype) ], dim=1)


		return predict_scores, best_cluster_idx, new_entity_memory, new_entity_size, cluster_loss 


if __name__ == '__main__':

	import utils.util as util
	config = util.initialize_from_env()

	from transformers import BartTokenizerFast
	tokenizer = BartTokenizerFast.from_pretrained(config['encoder_name'])
	
	torch.manual_seed(123)

	print("Preparing dummy input...")
	sample_sentences = [
		"The quick brown fox jumps ovaseqwe the lazy dog . Over A dog",
		"The quick bsadsyubsrown fox . Then again the same fox",
		"A brown fox jumps over .",
	]

	sample_spans = [
		[ {(0, 19), (20, 34)}, {(35, 47), (55, 60)} ],
		[ {(0, 27), (41, 53)} ],
		[ {(0, 6)}, {(12, 16)} ]
	]

	tokenized = tokenizer(sample_sentences, return_tensors="pt", padding=True, truncation=True, max_length=64, return_offsets_mapping=True)
	attn_mask = tokenized['attention_mask']
	input_ids = torch.randn(*tokenized['input_ids'].shape, config["hidden_size"])

	get_c2t_map = lambda offset: { x : i for i, (s, e) in enumerate(offset) for x in range(s, e) }
	c2t_map = list(map(get_c2t_map, tokenized['offset_mapping']))

	print("Gold spans: ")
	sample_span_tokenized = []
	mention_map = [ {} for _ in range(3) ]

	for x in range(3):
		sample_span_tokenized.append([])
		for c in sample_spans[x]:
			sample_span_tokenized[x].append(set())
			for s, e in c:
				t_s, t_e = c2t_map[x][s], (c2t_map[x][e-1] + 1) # span end index is exclusive
				sample_span_tokenized[x][-1].add((t_s, t_e))
				mention_map[x][(t_s, t_e)] = sample_span_tokenized[x][-1]

	print(sample_span_tokenized)

	print("Initializing the span encoder...")
	span_encoder = MentionEncoder(config)
	print(span_encoder)
	print("=" * 20)

	print("Initializing the memory module...")
	model = MemoryModule(config)
	print(model)
	print("=" * 20)

	opt = torch.optim.SGD([
				{'params': model.parameters()},
                {'params': span_encoder.parameters()}
			], lr=0.05)

	ctx = input_ids # ctx

	span_encoder.train()
	model.train()
	for i in range(1, 400):
		total_loss = 0
		embs, scrs, spns = span_encoder(ctx, attn_mask)
		predictions = [ [] for _ in range(3) ] # 3 is batch_size

		# forward pass in batches through time:
		e, es = None, None
		for t in range(spns.size(1)): # this is top k spans
			t_spans = [ tuple(spn.tolist()) for spn in spns[:, t] ] # spans in current step.

			g = []
			for batch in range(3):
				# if the relevant entity (cluster) of the current mention has been predicted before:
				if t_spans[batch] in mention_map[batch]:
					gold_entity_mentions = mention_map[batch]
					# g.append(some dynamic cluster id)
					g.append(1)
				else: # not a mention
					g.append(0)

			g = torch.LongTensor(g)
			s, b, e, es, l = model(ctx, embs[:, t], scrs[:, t], entity_memory=e, entity_size=es, gold_entity_ids=g)
			total_loss += l

			for batch in range(3):
				if b[batch] > 0:
					predictions[batch].append(t_spans[batch])

		total_loss.backward()
		opt.step()
		opt.zero_grad()

	print(predictions)
	print("=" * 20)
	print(total_loss)
	print("=" * 20)
