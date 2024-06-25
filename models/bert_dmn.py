from typing import Any, Dict, Iterable, List, Tuple, Union

import nltk
import torch
from ranking_utils.model import (
    PairwiseTrainingBatch,
    PointwiseTrainingBatch,
    Ranker,
    TrainingMode,
)
from ranking_utils.model.data import DataProcessor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.optim import AdamW
from transformers import BertModel, BertTokenizer, get_constant_schedule_with_warmup

from models.dmn import MemoryModule

BERTDMNInput = Tuple[List[int], List[int], torch.IntTensor]
BERTDMNBatch = Tuple[
    torch.LongTensor, torch.LongTensor, torch.LongTensor, torch.IntTensor
]


class BERTDMNDataProcessor(DataProcessor):
    """Data processor for Dynamic Memory Network rankers using BERT."""

    def __init__(self, bert_model: str, char_limit: int) -> None:
        """Constructor.

        Args:
            bert_model (str): Pre-trained BERT model.
            char_limit (int): Maximum number of characters per query/document.
        """
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)
        self.char_limit = char_limit

    def get_model_input(self, query: str, doc: str) -> BERTDMNInput:
        # empty queries or documents might cause problems later on
        if len(query.strip()) == 0:
            query = "(empty)"
        if len(doc.strip()) == 0:
            doc = "(empty)"

        query_tokenized = self.tokenizer.tokenize(query[: self.char_limit])
        doc_tokenized = []
        sentence_lengths = []
        for sentence in nltk.sent_tokenize(doc[: self.char_limit]):
            sentence_tokenized = self.tokenizer.tokenize(sentence)
            doc_tokenized.extend(sentence_tokenized)
            sentence_lengths.append(len(sentence_tokenized))
        return query_tokenized, doc_tokenized, torch.IntTensor(sentence_lengths)

    def get_model_batch(self, inputs: Iterable[BERTDMNInput]) -> BERTDMNBatch:
        queries_tokenized, docs_tokenized, sentence_lengths = zip(*inputs)
        inputs = self.tokenizer(
            queries_tokenized, docs_tokenized, padding=True, truncation=True
        )
        return (
            torch.LongTensor(inputs["input_ids"]),
            torch.LongTensor(inputs["attention_mask"]),
            torch.LongTensor(inputs["token_type_ids"]),
            pad_sequence(sentence_lengths, batch_first=True, padding_value=0),
        )


class InputModule(torch.nn.Module):
    """This module transforms BERT outputs into DMN inputs. It returns query and input representations
    (facts) after applying GRUs (input fusion).
    """

    def __init__(self, bert_dim: int, rep_dim: int, dropout: float) -> None:
        """Constructor.

        Args:
            bert_dim (int): Number of hidden BERT units.
            rep_dim (int): The dimension of fact and query representations.
            dropout (float): Dropout value.
        """
        super().__init__()
        self.rep_dim = rep_dim

        self.input_gru = torch.nn.GRU(
            bert_dim, rep_dim, bidirectional=True, batch_first=True
        )
        self.question_gru = torch.nn.GRU(bert_dim, rep_dim, batch_first=True)
        self.dropout = torch.nn.Dropout(dropout)

    def _forward_queries(
        self, queries: torch.Tensor, query_lengths: List[int]
    ) -> torch.Tensor:
        """Apply a GRU to a batch of queries.

        Args:
            queries (torch.Tensor): The query representations from BERT, each of shape (query_len, bert_dim).
            query_lengths (List[int]): The query lengths, shape (batch_size,).

        Returns:
            torch.Tensor: The last GRU outputs, shape (batch_size, 1, rep_dim).
        """
        queries_padded = pad_sequence(queries, batch_first=True)
        queries_packed = pack_padded_sequence(
            queries_padded, query_lengths, batch_first=True, enforce_sorted=False
        )
        _, h = self.question_gru(queries_packed)
        # transpose back to batch-first
        return torch.transpose(h, 0, 1)

    def _forward_facts(self, facts: torch.Tensor, num_facts: List[int]) -> torch.Tensor:
        """Apply dropout and a GRU to a batch of facts (input fusion).

        Args:
            facts (torch.Tensor): The fact representations from BERT, each of shape (num_facts, bert_dim).
            num_facts (List[int]): The number of facts, shape (batch_size,).

        Returns:
            torch.Tensor: The GRU output, shape (batch_size, max_facts_len, rep_dim).
        """
        facts_padded = pad_sequence(facts, batch_first=True)
        facts_packed = pack_padded_sequence(
            self.dropout(facts_padded),
            num_facts,
            batch_first=True,
            enforce_sorted=False,
        )
        gru_output, _ = self.input_gru(facts_packed)
        padded_output, _ = pad_packed_sequence(gru_output, batch_first=True)
        # sum outputs of both directions
        return padded_output[:, :, : self.rep_dim] + padded_output[:, :, self.rep_dim :]

    def forward(
        self, queries: List[torch.Tensor], facts: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
        """Extract the query representation and facts from BERT outputs and apply input fusion.

        Args:
            queries (List[torch.Tensor]): The query representations from BERT, each of shape (query_len, bert_dim).
            facts (List[torch.Tensor]): The fact representations from BERT, each of shape (num_facts, bert_dim).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]: A tuple containing
                * the query representations, shape (batch_size, 1, rep_dim),
                * the fact representations, shape (batch_size, max_num_facts, rep_dim),
                * the number of facts for each query, shape (batch_size,).
        """
        # for multi-gpu training
        self.input_gru.flatten_parameters()
        self.question_gru.flatten_parameters()

        query_lengths, num_facts = [], []
        for query_out, facts_out in zip(queries, facts):
            query_lengths.append(query_out.shape[0])
            num_facts.append(facts_out.shape[0])

        return (
            self._forward_queries(queries, query_lengths),
            self._forward_facts(facts, num_facts),
            num_facts,
        )


class AnswerModule(torch.nn.Module):
    """Answer module. Outputs a single relevance score for each input."""

    def __init__(self, dropout: float, bert_dim: int, rep_dim: int) -> None:
        """Constructor.

        Args:
            dropout (float): Dropout value.
            bert_dim (int): Number of hidden BERT units.
            rep_dim (int): The dimension of the memory.
        """
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout)
        self.linear = torch.nn.Linear(bert_dim + 2 * rep_dim, 1)

    def forward(
        self, cls_outputs: torch.Tensor, queries: torch.Tensor, memories: torch.Tensor
    ) -> torch.Tensor:
        """Return a batch of scores.

        Args:
            cls_outputs (torch.Tensor): The CLS outputs from BERT, shape (batch_size, bert_dim).
            queries (torch.Tensor): The query representations, shape (batch_size, rep_dim).
            memories (torch.Tensor): The final memories, shape (batch_size, rep_dim).

        Returns:
            torch.Tensor: The scores, shape (batch_size, 1).
        """
        return self.linear(
            self.dropout(torch.cat([cls_outputs, queries, memories], dim=1))
        )


class DMN(torch.nn.Module):
    """Dynamic Memory Network for BERT-based inputs."""

    def __init__(
        self,
        bert_dim: int,
        rep_dim: int,
        attention_dim: int,
        agru_dim: int,
        dropout: float,
        num_episodes: int,
    ) -> None:
        """Constructor.

        Args:
            bert_dim (int): Number of hidden BERT units.
            rep_dim (int): The dimension of fact and query representations and memory.
            attention_dim (int): The dimension of the linear layer applied to the interactions.
            agru_dim (int): The hidden dimension of the attention GRU.
            dropout (float): Dropout value.
            num_episodes (int): The number of DMN episodes.
        """
        super().__init__()
        self.input_module = InputModule(bert_dim, rep_dim, dropout)
        self.memory_module = MemoryModule(rep_dim, attention_dim, agru_dim)
        self.answer_module = AnswerModule(dropout, bert_dim, rep_dim)
        self.num_episodes = num_episodes

    def forward(
        self,
        cls_outputs: torch.Tensor,
        queries: List[torch.Tensor],
        facts: List[torch.Tensor],
    ) -> torch.Tensor:
        """Compute the relevance scores for a batch.

        Args:
            cls_outputs (torch.Tensor): BERT outputs corresponding to the CLS tokens, shape (batch_size, bert_dim).
            queries (List[torch.Tensor]): The query representations from BERT, each of shape (query_len, bert_dim).
            facts (List[torch.Tensor]): The fact representations from BERT, each of shape (num_facts, bert_dim).

        Returns:
            torch.Tensor: The scores, shape (batch_size, 1).
        """
        queries, facts, num_facts = self.input_module(queries, facts)
        m = queries
        for _ in range(self.num_episodes):
            m = self.memory_module(queries, facts, num_facts, m)
        return self.answer_module(cls_outputs, queries.squeeze(1), m.squeeze(1))


class BERTDMNRanker(Ranker):
    """Dynamic Memory Network for ranking using BERT."""

    def __init__(
        self,
        lr: float,
        warmup_steps: int,
        hparams: Dict[str, Any],
    ) -> None:
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

        self.cache, self.pos_cache, self.neg_cache = {}, {}, {}
        self.bert = BertModel.from_pretrained(hparams["bert_model"], return_dict=True)

        self.dmn = DMN(
            self.bert.encoder.layer[-1].output.dense.out_features,
            hparams["rep_dim"],
            hparams["attention_dim"],
            hparams["agru_dim"],
            hparams["dropout"],
            hparams["num_episodes"],
        )
        self.sep_id = BertTokenizer.from_pretrained(hparams["bert_model"]).sep_token_id

        if hparams["lite"]:
            for p in self.bert.parameters():
                p.requires_grad = False

    def forward(
        self,
        batch: BERTDMNBatch,
        data_indices: torch.Tensor = None,
        cache: Dict[int, torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the relevance scores for a batch.

        Args:
            batch (BERTDMNBatch): BERT-DMN inputs.
            data_indices (torch.Tensor, optional): Unique data IDs used for caching. Defaults to None.
            cache (Dict[int, torch.Tensor], optional): Maps IDs to cached output tensors. Defaults to None.

        Returns:
            torch.Tensor: The scores, shape (batch_size, 1).
        """
        in_ids, masks, tt_ids, sentence_lengths = batch
        cls_outputs, query_outputs, fact_outputs = [], [], []

        # BERT-DMN-lite caching
        if self.hparams["lite"] and not self.hparams["no_cache"] and self.training:
            temp_outputs = {}
            missing_idxs = []
            missing_in_ids = []
            missing_masks = []
            missing_tt_ids = []
            missing_sentence_lengths = []

            # the inputs have a unique ID each
            for i, idx in enumerate(data_indices):
                idx = int(idx)

                # if the output for the ID was already cached, put it back on this device
                if idx in cache:
                    temp_outputs[idx] = [t.to(in_ids.device) for t in cache[idx]]

                # all missing outputs will be computed in one batch
                else:
                    missing_idxs.append(idx)
                    missing_in_ids.append(in_ids[i])
                    missing_masks.append(masks[i])
                    missing_tt_ids.append(tt_ids[i])
                    missing_sentence_lengths.append(sentence_lengths[i])

            # compute missing outputs if there are any and put them in the cache
            if len(missing_idxs) > 0:
                missing_in_ids = torch.stack(missing_in_ids)
                missing_masks = torch.stack(missing_masks)
                missing_tt_ids = torch.stack(missing_tt_ids)
                missing_sentence_lengths = torch.stack(missing_sentence_lengths)

                last_hidden_state = self.bert(
                    missing_in_ids, missing_masks, missing_tt_ids
                )["last_hidden_state"]
                for idx, item_in, item_out, item_sentence_lengths in zip(
                    missing_idxs,
                    missing_in_ids,
                    last_hidden_state,
                    missing_sentence_lengths,
                ):
                    cls_out, query_out, facts_out = self._split_outputs(
                        item_in, item_out, item_sentence_lengths
                    )
                    temp_outputs[idx] = cls_out, query_out, facts_out
                    cache[idx] = cls_out.cpu(), query_out.cpu(), facts_out.cpu()

            # assemble all outputs in the correct order
            for idx in data_indices:
                cls_out, query_out, facts_out = temp_outputs[int(idx)]
                cls_outputs.append(cls_out)
                query_outputs.append(query_out)
                fact_outputs.append(facts_out)

        # inference/regular BERT-DMN training
        else:
            last_hidden_state = self.bert(in_ids, masks, tt_ids)["last_hidden_state"]
            for item_in, item_out, item_sentence_lengths in zip(
                in_ids, last_hidden_state, sentence_lengths
            ):
                cls_out, query_out, facts_out = self._split_outputs(
                    item_in, item_out, item_sentence_lengths
                )
                cls_outputs.append(cls_out)
                query_outputs.append(query_out)
                fact_outputs.append(facts_out)

        return self.dmn(torch.stack(cls_outputs), query_outputs, fact_outputs)

    def _split_outputs(
        self,
        item_in: torch.Tensor,
        item_out: torch.Tensor,
        item_sentence_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split a BERT output sequence into query and fact outputs.

        Args:
            item_in (torch.Tensor): BERT input IDs, shape (seq_len,).
            item_out (torch.Tensor): Corresponding BERT outputs, shape (seq_len, bert_dim).
            item_sentence_lengths (torch.Tensor): Sentence lengths, shape (max_num_sent).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing
                * the BERT outputs for the CLS token, shape (bert_dim,),
                * the query outputs, shape (query_len, bert_dim),
                * the document facts as averaged sentence outputs, shape (num_sentences, bert_dim).
        """
        # we always have 2 separators
        sep1, sep2 = (item_in == self.sep_id).nonzero(as_tuple=False).squeeze()

        # split outputs into query and document parts
        query_outputs = item_out[1:sep1]
        doc_outputs = item_out[sep1 + 1 : sep2]

        # split the document outputs into facts
        fact_outputs = []
        idx = 0
        # we will get all sentences this way, even if the last one was truncated
        for length in item_sentence_lengths:
            s = doc_outputs[idx : idx + length]
            idx += length
            if len(s) > 0:
                fact_outputs.append(torch.mean(s, dim=0))
        return item_out[0], query_outputs, torch.stack(fact_outputs)

    def configure_optimizers(self) -> Tuple[List[Any], List[Any]]:
        params = [{"params": self.dmn.parameters(), "lr": self.lr}]
        if not self.hparams["lite"]:
            params.append({"params": self.bert.parameters(), "lr": self.lr})
        opt = AdamW(params)
        sched = get_constant_schedule_with_warmup(opt, self.warmup_steps)
        return [opt], [{"scheduler": sched, "interval": "step"}]

    def training_step(
        self,
        batch: Union[PointwiseTrainingBatch, PairwiseTrainingBatch],
        batch_idx: int,
    ) -> torch.Tensor:
        if self.training_mode == TrainingMode.POINTWISE:
            model_batch, labels, indices = batch
            outputs = torch.sigmoid(self(model_batch, indices, self.cache))
            loss = self.bce(outputs.flatten(), labels.flatten())
        elif self.training_mode == TrainingMode.PAIRWISE:
            pos_inputs, neg_inputs, indices = batch
            pos_outputs = torch.sigmoid(self(pos_inputs, indices, self.pos_cache))
            neg_outputs = torch.sigmoid(self(neg_inputs, indices, self.neg_cache))
            loss = torch.mean(
                torch.clamp(
                    self.pairwise_loss_margin - pos_outputs + neg_outputs, min=0
                )
            )

        self.log("train_loss", loss)
        return loss
