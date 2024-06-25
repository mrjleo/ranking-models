import abc
from typing import Any, Dict, Iterable, List, Tuple

import nltk
import torch
from ranking_utils.model import Ranker
from ranking_utils.model.data import DataProcessor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.optim import AdamW
from transformers import BertModel, BertTokenizer, get_constant_schedule_with_warmup

SRInput = Tuple[torch.LongTensor, torch.LongTensor, torch.IntTensor]
SRBatch = Tuple[
    torch.LongTensor,
    torch.IntTensor,
    torch.LongTensor,
    torch.IntTensor,
    torch.IntTensor,
]


class SRDataProcessor(DataProcessor):
    """Data processor for Select & Rank models."""

    def __init__(
        self,
        bert_model: str,
        max_query_tokens: int,
        max_doc_tokens: int,
        max_sentences: int,
        passage_length: int,
    ) -> None:
        """Constructor.

        Args:
            bert_model (str): Pre-trained BERT model.
            max_query_tokens (int): Maximum number of query tokens.
            max_doc_tokens (int): Maximum number of document tokens.
            max_sentences (int): Maximum number of sentences considered in a document.
            passage_length (int): Number of sentences per passage.
        """
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)
        self.max_query_tokens = max_query_tokens
        self.max_doc_tokens = max_doc_tokens
        self.max_sentences = max_sentences
        self.passage_length = passage_length

    def get_model_input(self, query: str, doc: str) -> SRInput:
        # empty queries or documents might cause problems later on
        if len(query.strip()) == 0:
            query = "(empty)"
        if len(doc.strip()) == 0:
            doc = "(empty)"

        query_inputs = self.tokenizer(query, add_special_tokens=False)["input_ids"][
            : self.max_query_tokens
        ]
        assert len(query_inputs) <= self.max_query_tokens

        doc_inputs = []
        lengths = []
        tokens = nltk.sent_tokenize(doc)[: self.max_sentences]
        for i in range(0, len(tokens), self.passage_length):
            passage = " ".join(tokens[i : i + self.passage_length])
            inputs = self.tokenizer(passage, add_special_tokens=False)["input_ids"]

            if sum(lengths) + len(inputs) > self.max_doc_tokens:
                remaining = self.max_doc_tokens - sum(lengths)
                doc_inputs.extend(inputs[:remaining])
                lengths.append(remaining)
                break
            else:
                doc_inputs.extend(inputs)
                lengths.append(len(inputs))

        assert len(doc_inputs) == sum(lengths)
        assert len(doc_inputs) <= self.max_doc_tokens
        return (
            torch.LongTensor(query_inputs),
            torch.LongTensor(doc_inputs),
            torch.IntTensor(lengths),
        )

    def get_model_batch(self, inputs: Iterable[SRInput]) -> SRBatch:
        query_inputs, doc_inputs, lengths = zip(*inputs)
        query_lengths = torch.IntTensor(list(map(len, query_inputs)))
        doc_lengths = torch.IntTensor(list(map(len, doc_inputs)))
        query_inputs_padded = pad_sequence(
            query_inputs, batch_first=True, padding_value=0
        )
        doc_inputs_padded = pad_sequence(doc_inputs, batch_first=True, padding_value=0)
        lengths_padded = pad_sequence(lengths, batch_first=True, padding_value=0)
        return (
            query_inputs_padded,
            query_lengths,
            doc_inputs_padded,
            doc_lengths,
            lengths_padded,
        )


class AttentionSelector(torch.nn.Module):
    """Attention-based LSTM selector. Query and document are encoded using a shared bidirectional LSTM.
    Each passage in the document is scored based on cosine similarity to the query.
    """

    def __init__(
        self,
        embedding: torch.nn.Embedding,
        input_dim: int,
        hidden_dim: int,
        attention_dim: int,
        dropout: float,
    ) -> None:
        """Constructor.

        Args:
            embedding (Embedding): Embedding for query and document tokens.
            input_dim (int): Input dimension.
            hidden_dim (int): LSTM hidden dimension.
            attention_dim (int): Attention dimension.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.embedding = embedding
        self.lstm = torch.nn.LSTM(
            input_dim, hidden_dim, batch_first=True, bidirectional=True
        )

        # attention
        self.W_attn_q = torch.nn.Linear(hidden_dim * 2, attention_dim)
        self.W_attn_s = torch.nn.Linear(hidden_dim * 2, attention_dim)
        self.w_attn = torch.nn.Linear(attention_dim, 1)
        self.tanh = torch.nn.Tanh()
        self.softmax_attn = torch.nn.Softmax(dim=1)

        self.dropout = torch.nn.Dropout(dropout)
        self.cos_sim = torch.nn.CosineSimilarity(dim=1)

    def encode_batch(
        self, sequences: torch.LongTensor, lengths: torch.IntTensor
    ) -> torch.Tensor:
        """Encode a batch of sequences using the LSTM and apply max pooling.

        Args:
            sequences (torch.LongTensor): A batch of sequences.
            lengths (torch.IntTensor): Sequence lengths.

        Returns:
            torch.Tensor: The sequence representations.
        """
        seq_enc = self._encode(sequences, lengths)
        return self._max_pool(seq_enc, lengths)

    def _encode(
        self, inputs: torch.LongTensor, lengths: torch.IntTensor
    ) -> torch.Tensor:
        """Encode sequences using the LSTM.

        Args:
            inputs (torch.LongTensor): Input IDs.
            lengths (torch.IntTensor): Input lengths.

        Returns:
            torch.Tensor: The LSTM outputs.
        """
        inputs_emb = self.embedding(inputs)
        inputs_packed = pack_padded_sequence(
            inputs_emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        lstm_out, _ = self.lstm(inputs_packed)
        out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        return out

    def _max_pool(
        self, inputs: torch.FloatTensor, lengths: torch.IntTensor
    ) -> torch.FloatTensor:
        """Masked max pooling.

        Args:
            inputs (torch.FloatTensor): Input sequences.
            lengths (torch.IntTensor): Sequence lengths.

        Returns:
            torch.FloatTensor: Output tensor.
        """
        num_items, max_len, num_hidden = inputs.shape

        # create mask
        rng = (
            torch.arange(max_len, device=inputs.device)
            .unsqueeze(0)
            .expand(num_items, -1)
        )
        rng = rng.unsqueeze(-1).expand(-1, -1, num_hidden)
        lengths = lengths.unsqueeze(1).expand(-1, num_hidden)
        lengths = lengths.unsqueeze(1).expand(-1, max_len, -1)
        mask = rng < lengths[:num_items]

        # set padding outputs to -inf so they dont affect max pooling
        inputs_clone = inputs.clone()
        inputs_clone[~mask] = float("-inf")
        return torch.max(inputs_clone, 1)[0]

    def _get_passage_outputs(
        self, doc_encoded: torch.FloatTensor, lengths: torch.IntTensor
    ) -> List[torch.FloatTensor]:
        """Split the outputs of a single document into passages.

        Args:
            doc_encoded (torch.FloatTensor): The LSTM document outputs.
            lengths (torch.IntTensor): The passage lengths.

        Returns:
            torch.FloatTensor: The passage outputs, padded.
        """
        result = []
        idx = 0
        for length in lengths:

            # if the length is zero, we reached the padding
            if length == 0:
                break

            next_idx = idx + length
            result.append(doc_encoded[idx:next_idx])
            idx = next_idx
        return pad_sequence(result, batch_first=True)

    def _attention(
        self,
        query_rep_attn: torch.FloatTensor,
        outputs: torch.FloatTensor,
        lengths: torch.IntTensor,
    ) -> torch.FloatTensor:
        """Compute a simple query-passage attention.

        Args:
            query_rep_attn (torch.FloatTensor): Query representation (pre-computed attention).
            outputs (torch.FloatTensor): Outputs for each passage in a document.
            lengths (torch.IntTensor): Passage lengths.

        Returns:
            torch.FloatTensor: Results to be pooled and used as passage representations.
        """
        num, max_len, _ = outputs.shape
        attn = self.W_attn_s(outputs)
        attn_sum = query_rep_attn.expand_as(attn) + attn
        attn_weights = self.w_attn(self.tanh(attn_sum))

        # mask the padding tokens before computing the softmax by setting the corresponding values to -inf
        mask = (
            torch.arange(max_len, device=query_rep_attn.device)[None, :]
            < lengths[:num, None]
        )
        attn_weights[~mask] = float("-inf")
        return outputs * self.softmax_attn(attn_weights)

    def _get_scores(
        self, query_rep: torch.FloatTensor, passage_reps: torch.FloatTensor
    ) -> torch.FloatTensor:
        """Return scores between query and each passage using cosine similarity, applying dropout.

        Args:
            query_rep (torch.FloatTensor): Query representation.
            passage_reps (torch.FloatTensor): Passage representations.

        Returns:
            torch.FloatTensor: A score for each passage.
        """
        a = query_rep.expand_as(passage_reps)
        b = passage_reps
        return self.cos_sim(self.dropout(a), self.dropout(b))

    def forward(self, batch: SRBatch) -> List[torch.FloatTensor]:
        """Return passage scores.

        Args:
            batch (SRBatch): Input batch.

        Returns:
            List[torch.FloatTensor]: A score for each passage in each input sequence.
        """
        q_in, q_lens, d_in, d_lens, p_lens = batch

        # queries and docs are encoded using the shared LSTM
        q_enc = self._encode(q_in, q_lens)
        d_enc = self._encode(d_in, d_lens)

        # queries are represented as the average of the LSTM outputs
        q_reps = self._max_pool(q_enc, q_lens)

        # pre-compute query representations for the attention
        q_reps_att = self.W_attn_q(q_reps)

        scores = []
        for q_rep, q_rep_att, d_enc_, p_lens_ in zip(q_reps, q_reps_att, d_enc, p_lens):
            p_out = self._get_passage_outputs(d_enc_, p_lens_)
            p_out_attn = self._attention(q_rep_att, p_out, p_lens_)
            p_reps = self._max_pool(p_out_attn, p_lens_)
            scores.append(self._get_scores(q_rep, p_reps))
        return scores


class LinearSelector(torch.nn.Module):
    """Linear dot-product-based selector. Query and document embeddings are averaged and encoded using a linear layer.
    Each passage in the document is scored using a dot product with the query.
    """

    def __init__(
        self,
        embedding: torch.nn.Embedding,
        input_dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        """Constructor.

        Args:
            embedding (Embedding): Embedding for query and document tokens.
            input_dim (int): Input dimension.
            hidden_dim (int): Hidden dimension.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.embedding = embedding
        self.linear = torch.nn.Linear(input_dim, hidden_dim)
        self.dropout = torch.nn.Dropout(dropout)

    def encode_batch(
        self, sequences: torch.LongTensor, lengths: torch.IntTensor
    ) -> torch.Tensor:
        """Encode a batch of sequences, each as the average of its embeddings, and apply the linear layer.

        Args:
            sequences (torch.LongTensor): A batch of sequences.
            lengths (torch.IntTensor): Sequence lengths.

        Returns:
            torch.Tensor: The sequence representations.
        """
        # embed all sequences
        sequences_emb = self.embedding(sequences)

        # create a mask corresponding to sequence lengths
        _, max_len, emb_dim = sequences_emb.shape
        mask = torch.arange(max_len, device=lengths.device).unsqueeze(
            0
        ) < lengths.unsqueeze(-1)
        mask = mask.unsqueeze(-1).expand(-1, -1, emb_dim)

        # compute the mean for each passage
        passage_reps = torch.sum(mask * sequences_emb, dim=1) / lengths.unsqueeze(-1)
        return self.linear(passage_reps)

    def _get_encoded_passages(
        self, doc_inputs: torch.FloatTensor, lengths: torch.IntTensor
    ) -> List[torch.FloatTensor]:
        """Split a document into passages and encode them.

        Args:
            doc_inputs (torch.FloatTensor): The document input IDs.
            lengths (torch.IntTensor): The passage lengths.

        Returns:
            torch.FloatTensor: The encoded passages.
        """
        passages = []
        idx = 0
        for length in lengths:

            # if the length is zero, we reached the padding
            if length == 0:
                break

            next_idx = idx + length
            passages.append(doc_inputs[idx:next_idx])
            idx = next_idx

        passages_padded = pad_sequence(passages, batch_first=True, padding_value=0)
        num_passages = passages_padded.shape[0]
        return self.encode_batch(passages_padded, lengths[:num_passages])

    def _get_scores(
        self, query_encoded: torch.FloatTensor, passages_encoded: torch.FloatTensor
    ) -> torch.FloatTensor:
        """Return all scores between a query and each corresponding passage using a dot product, applying dropout.

        Args:
            query_encoded (torch.FloatTensor): Query representation.
            passages_encoded (torch.FloatTensor): Passage representations.

        Returns:
            torch.FloatTensor: A score for each passage.
        """
        # row-wise dot product with individual dropout
        a = query_encoded.expand_as(passages_encoded)
        b = passages_encoded
        return (self.dropout(a) * self.dropout(b)).sum(-1)

    def forward(self, batch: SRBatch) -> List[torch.FloatTensor]:
        """Return passage scores.

        Args:
            batch (SRBatch): Input batch

        Returns:
            List[torch.FloatTensor]: A score for each passage in each input sequence.
        """
        queries, query_lengths, docs, _, passage_lengths = batch
        queries_encoded = self.encode_batch(queries, query_lengths)
        scores = []
        for query_encoded, doc, passage_lengths_item in zip(
            queries_encoded, docs, passage_lengths
        ):
            passages_encoded = self._get_encoded_passages(doc, passage_lengths_item)
            scores.append(self._get_scores(query_encoded, passages_encoded))
        return scores


class BERTRanker(torch.nn.Module):
    """A simple BERT ranker that uses the CLS output as a score. The inputs are constructed using an approximated
    k-hot sample, i.e. only k sentences are considered. The correspondign inputs are multiplied with their weights.
    """

    def __init__(self, bert_model: str, dropout: float, freeze: bool = False) -> None:
        """Constructor.

        Args:
            bert_model (str): Pre-trained BERT model.
            dropout (float): Dropout value.
            freeze (bool, optional): Don't update ranker weights during training. Defaults to False.
        """
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model, return_dict=True)
        self.dropout = torch.nn.Dropout(dropout)
        self.classification = torch.nn.Linear(
            self.bert.encoder.layer[-1].output.dense.out_features, 1
        )

        tokenizer = BertTokenizer.from_pretrained(bert_model)
        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id
        self.pad_id = tokenizer.pad_token_id
        self.max_len = 512

        for p in self.parameters():
            p.requires_grad = not freeze

    def _get_single_input(
        self,
        query_in: torch.LongTensor,
        doc_in: torch.LongTensor,
        lengths: torch.IntTensor,
        weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Construct a single BERT input sequence.

        Args:
            query_in (torch.LongTensor): Query input IDs.
            doc_in (torch.LongTensor): Document input IDs.
            lengths (Sequence[int]): Passage lengths.
            weights (torch.Tensor): Passage weights.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: BERT inputs and weights.
        """
        # device for new tensors
        dev = query_in.device
        cls_tensor = torch.as_tensor([self.cls_id], device=dev)
        sep_tensor = torch.as_tensor([self.sep_id], device=dev)

        # keep track of sequence length to construct padding, mask and token type IDs
        in_ids = [cls_tensor, query_in, sep_tensor]
        running_seq_len = len(query_in) + 2

        # token types are 0 up until (including) the 1st SEP
        tt_ids = [torch.as_tensor([0] * running_seq_len, device=dev)]

        # CLS/query part -- set all weights to 1, as we always keep the query
        in_weights = [torch.as_tensor([1.0] * running_seq_len, device=dev)]

        # document part -- drop passages with weight of 0, copy the weights for all
        idx = 0
        for length, weight in zip(lengths, weights):
            # we should never see padding here as the number of weights is equal to the number of passages
            assert length > 0

            next_idx = idx + length
            if weight == 1.0:
                in_ids.append(doc_in[idx:next_idx])

                # token types are 1 up until (including) the 2nd SEP
                tt_ids.append(torch.as_tensor([1] * length, device=dev))

                in_weights.extend([weight.unsqueeze(0)] * length)
                running_seq_len += length
            idx = next_idx

        # last token should be SEP
        in_ids.append(sep_tensor)
        running_seq_len += 1

        # mask is 1 up until the 2nd SEP
        mask = [torch.as_tensor([1.0] * running_seq_len, device=dev)]

        # if the sequence is not max length, pad
        remaining = self.max_len - min(running_seq_len, self.max_len)
        if remaining > 0:
            in_ids.append(torch.as_tensor([self.pad_id] * remaining, device=dev))

            # token types and mask are 0 after the 2st SEP
            mask.append(torch.as_tensor([0.0] * remaining, device=dev))

        # these need one more for the separator
        tt_ids.append(torch.as_tensor([0] * (remaining + 1), device=dev))
        in_weights.append(torch.as_tensor([1.0] * (remaining + 1), device=dev))

        # truncate to maximum length
        in_ids = torch.cat(in_ids)[: self.max_len]
        mask = torch.cat(mask)[: self.max_len]
        tt_ids = torch.cat(tt_ids)[: self.max_len]
        in_weights = torch.cat(in_weights)[: self.max_len]

        # make sure lengths match
        assert (
            in_ids.shape[-1]
            == mask.shape[-1]
            == tt_ids.shape[-1]
            == in_weights.shape[-1]
            == self.max_len
        )
        return in_ids, mask, tt_ids, in_weights

    def forward(self, batch: SRBatch, weights: torch.Tensor) -> torch.FloatTensor:
        """Classify a batch of inputs, using the k highest scored passages as BERT input.

        Args:
            batch (SRBatch): The input batch.
            weights (torch.Tensor): The weights.

        Returns:
            torch.FloatTensor: Relevance scores for each input.
        """
        batch_in_ids, batch_masks, batch_tt_ids, batch_weights = [], [], [], []
        for query_in, query_length, doc_in, doc_length, lengths, weights_ in zip(
            *batch, weights
        ):

            # remove padding
            query_in = query_in[:query_length]
            doc_in = doc_in[:doc_length]

            # create BERT inputs
            in_ids, mask, tt_ids, weights_in = self._get_single_input(
                query_in, doc_in, lengths, weights_
            )
            batch_in_ids.append(in_ids)
            batch_masks.append(mask)
            batch_tt_ids.append(tt_ids)
            batch_weights.append(weights_in)

        # create a batch of BERT inputs
        batch_in_ids = torch.stack(batch_in_ids)
        batch_masks = torch.stack(batch_masks)
        batch_tt_ids = torch.stack(batch_tt_ids)
        batch_weights = torch.stack(batch_weights)

        # create actual input by multiplying weights with input embeddings
        batch_emb = self.bert.embeddings(input_ids=batch_in_ids)
        bert_in_batch = batch_emb * batch_weights.unsqueeze(-1).expand_as(batch_emb)

        bert_out = self.bert(
            inputs_embeds=bert_in_batch,
            token_type_ids=batch_tt_ids,
            attention_mask=batch_masks,
        )
        cls_out = bert_out["last_hidden_state"][:, 0]
        return self.classification(self.dropout(cls_out))


class SRBase(Ranker, abc.ABC):
    """Base class for Select & Rank models. Each passage is assigned a score by the selector.
    A weighted subset sample of size k is drawn to select the input passages for the ranker.

    Methods to be implemented:
        * get_ranker
        * get_selector
    """

    def __init__(self, lr: int, warmup_steps: int, hparams: Dict[str, Any]) -> None:
        """Constructor.

        Args:
            lr (float): Learning rate.
            warmup_steps (int): Number of warmup steps.
            hparams (Dict[str, Any]): Hyperparameters.
        """
        super().__init__()
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.save_hyperparameters(hparams)

        # used for sampling
        self.eps = torch.finfo(torch.float32).eps

    @abc.abstractmethod
    def get_selector(self) -> Any:
        """Return the selector.

        Returns:
            Any: The selector.
        """
        pass

    @abc.abstractmethod
    def get_ranker(self) -> Any:
        """Return the ranker.

        Returns:
            Any: The ranker.
        """
        pass

    def _sample_subset(self, scores: torch.FloatTensor) -> torch.Tensor:
        """Create a subset as a k-hot vector using Gumbel-softmax sampling.
        The gradients are preserved by the straight-through trick.

        Args:
            scores (torch.FloatTensor): Scores output by the selector.

        Returns:
            torch.Tensor: A k-hot subset sample.
        """
        probs = torch.nn.functional.softmax(scores, dim=0)
        log_probs = torch.log(probs)

        # gumbel softmax sampling
        U = torch.rand(log_probs.shape, device=log_probs.device)
        r_hat = log_probs - torch.log(-torch.log(U + self.eps) + self.eps)

        # there might be less than k passages
        k = min(self.hparams["k"], r_hat.shape[-1])

        # relaxed top-k procedure
        t = self.hparams["temperature"]
        p = []
        alpha = r_hat
        for _ in range(k):
            p_j = torch.nn.functional.softmax(alpha / t, dim=0)
            p.append(p_j)
            alpha += torch.log(1 - p_j)

        # the approximated k-hot vector
        v = torch.stack(p).sum(dim=0)

        # exact k-hot vector
        _, topk_indices = v.topk(k)
        k_hot = torch.zeros_like(v).scatter_(0, topk_indices, 1)

        # return the exact k-hot vector with soft gradients (straight-through)
        return (k_hot - v).detach() + v

    def _get_topk(self, scores: torch.FloatTensor) -> torch.Tensor:
        """Return a k-hot vector with the top passages, without sampling or gradients.

        Args:
            scores (torch.FloatTensor): Scores output by the selector.

        Returns:
            torch.Tensor: A k-hot vector.
        """
        # there might be less than k passages
        k = min(self.hparams["k"], scores.shape[-1])
        _, topk_indices = scores.topk(k)
        return torch.zeros_like(scores).scatter_(0, topk_indices, 1)

    def forward(self, batch: SRBatch) -> torch.Tensor:
        """Forward pass. Return scores for a batch of inputs.

        Args:
            batch (SRBatch): The input batch.

        Returns:
            torch.Tensor: The scores.
        """
        scores = self.get_selector()(batch)
        # during training we sample, during inference we take the best k
        weights = [
            self._sample_subset(s) if self.training else self._get_topk(s)
            for s in scores
        ]
        return self.get_ranker()(batch, weights)

    def configure_optimizers(self) -> Tuple[List[Any], List[Any]]:
        params_with_grad = filter(lambda p: p.requires_grad, self.parameters())
        opt = AdamW(params_with_grad, lr=self.lr)
        sched = get_constant_schedule_with_warmup(opt, self.warmup_steps)
        return [opt], [{"scheduler": sched, "interval": "step"}]


class SelectAndRankAttention(SRBase):
    """Select & Rank model that uses an LSTM selector with attention and a BERT ranker."""

    def __init__(self, lr: int, warmup_steps: int, hparams: Dict[str, Any]) -> None:
        """Constructor.

        Args:
            lr (float): Learning rate.
            warmup_steps (int): Number of warmup steps.
            hparams (Dict[str, Any]): Hyperparameters.
        """
        super().__init__(lr, warmup_steps, hparams)
        self.ranker = BERTRanker(
            hparams["bert_model"],
            hparams["dropout"],
            hparams["freeze_ranker"],
        )

        # we re-use the embedding of the ranker so the selector has the same vocabulary
        self.selector = AttentionSelector(
            self.ranker.bert.get_input_embeddings(),
            self.ranker.bert.encoder.layer[-1].output.dense.out_features,
            hparams["lstm_dim"],
            hparams["attention_dim"],
            hparams["dropout"],
        )

    def get_selector(self) -> AttentionSelector:
        return self.selector

    def get_ranker(self) -> BERTRanker:
        return self.ranker


class SelectAndRankLinear(SRBase):
    """Select & Rank model with a linear selector and BERT ranker."""

    def __init__(self, lr: int, warmup_steps: int, hparams: Dict[str, Any]) -> None:
        """Constructor.

        Args:
            lr (float): Learning rate.
            warmup_steps (int): Number of warmup steps.
            hparams (Dict[str, Any]): Hyperparameters.
        """
        super().__init__(lr, warmup_steps, hparams)
        self.ranker = BERTRanker(
            hparams["bert_model"],
            hparams["dropout"],
            hparams["freeze_ranker"],
        )

        # we re-use the embedding of the ranker so the selector has the same vocabulary
        self.selector = LinearSelector(
            self.ranker.bert.get_input_embeddings(),
            self.ranker.bert.encoder.layer[-1].output.dense.out_features,
            hparams["hidden_dim"],
            hparams["dropout"],
        )

    def get_selector(self) -> LinearSelector:
        return self.selector

    def get_ranker(self) -> BERTRanker:
        return self.ranker
