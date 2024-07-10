from typing import Any, Dict, Iterable, List, Tuple

import nltk
import torch
from ranking_utils.model import Ranker
from ranking_utils.model.data import DataProcessor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.optim import Adam
from torchtext.vocab import Vectors
from transformers import get_constant_schedule_with_warmup

DMNInput = Tuple[torch.LongTensor, torch.LongTensor, torch.IntTensor]
DMNBatch = Tuple[torch.LongTensor, torch.IntTensor, torch.LongTensor, torch.LongTensor]


class DMNDataProcessor(DataProcessor):
    """Data processor for Dynamic Memory Network rankers."""

    def __init__(self, embeddings: Vectors) -> None:
        """Instantiate a data processor for DMN rankers.

        Args:
            embeddings (Vectors): Pre-trained embedding vectors (torchtext).
        """
        super().__init__()
        self.stoi = embeddings.stoi
        self.unk_id = len(self.stoi)
        self.pad_id = len(self.stoi) + 1

    def get_model_input(self, query: str, doc: str) -> DMNInput:
        # empty queries or documents might cause problems later on
        if len(query.strip()) == 0:
            query = "(empty)"
        if len(doc.strip()) == 0:
            doc = "(empty)"

        query_tokens = [
            self.stoi.get(w, self.unk_id) for w in nltk.word_tokenize(query)
        ]
        doc_tokens = []
        sentence_lengths = []
        for sentence in nltk.sent_tokenize(doc):
            sentence_tokens = [
                self.stoi.get(w, self.unk_id) for w in nltk.word_tokenize(sentence)
            ]
            doc_tokens.extend(sentence_tokens)
            sentence_lengths.append(len(sentence_tokens))
        return (
            torch.LongTensor(query_tokens),
            torch.LongTensor(doc_tokens),
            torch.IntTensor(sentence_lengths),
        )

    def get_model_batch(self, inputs: Iterable[DMNInput]) -> DMNBatch:
        batch_query_tokens, batch_doc_tokens, batch_sentence_lengths = zip(*inputs)
        query_lengths = [len(x) for x in batch_query_tokens]
        return (
            pad_sequence(
                batch_query_tokens, batch_first=True, padding_value=self.pad_id
            ),
            torch.IntTensor(query_lengths),
            pad_sequence(batch_doc_tokens, batch_first=True, padding_value=self.pad_id),
            pad_sequence(batch_sentence_lengths, batch_first=True, padding_value=0),
        )


class InputModule(torch.nn.Module):
    """This module computes the query and fact representations."""

    def __init__(self, embeddings: Vectors, rep_dim: int, dropout: float) -> None:
        """Instantiate a DMN input module.

        Args:
            embeddings (Vectors): Pre-trained embedding vectors (torchtext).
            rep_dim (int): The dimension of the fact representations.
            dropout (float): Dropout value.
        """
        super().__init__()
        self.rep_dim = rep_dim

        # add <unk> and <pad>
        num_embeddings = len(embeddings.vectors) + 2
        emb_dim = embeddings.vectors[0].shape[0]
        self.embedding = torch.nn.Embedding(
            num_embeddings, emb_dim, padding_idx=len(embeddings.vectors) + 1
        )

        # load pre-trained embeddings
        with torch.no_grad():
            self.embedding.weight[0 : len(embeddings.vectors)] = embeddings.vectors

        self.input_gru = torch.nn.GRU(
            emb_dim, rep_dim, bidirectional=True, batch_first=True
        )
        self.question_gru = torch.nn.GRU(emb_dim, rep_dim, batch_first=True)
        self.dropout = torch.nn.Dropout(dropout)

    def _get_facts(
        self, embedded_doc: torch.Tensor, item_sentence_lengths: torch.Tensor
    ) -> torch.Tensor:
        """Return all facts from a document.

        Args:
            embedded_doc (torch.Tensor): The embedded document, shape (doc_len, emb_dim).
            item_sentence_lengths (torch.Tensor): Sentence lengths, shape (max_num_sent,).

        Returns:
            torch.Tensor: The facts, shape (num_facts, emb_dim).
        """
        facts = []
        idx = 0
        # we will get all sentences this way, even if the last one was truncated
        for length in item_sentence_lengths:
            s = embedded_doc[idx : idx + length]
            idx += length
            if len(s) > 0:
                facts.append(torch.mean(s, dim=0))
        return torch.stack(facts)

    def _forward_queries(
        self, queries: torch.Tensor, query_lengths: List[int]
    ) -> torch.Tensor:
        """Apply a GRU to a batch of queries.

        Args:
            queries (torch.Tensor): The query representations, shape (batch_size, max_query_len, glove_dim).
            query_lengths (List[int]): The query lengths, shape (batch_size,).

        Returns:
            torch.Tensor: The last GRU outputs, shape (batch_size, 1, rep_dim).
        """
        queries_packed = pack_padded_sequence(
            queries, query_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.question_gru(queries_packed)
        # transpose back to batch-first
        return torch.transpose(h, 0, 1)

    def _forward_facts(self, facts: torch.Tensor, num_facts: List[int]) -> torch.Tensor:
        """Apply dropout and a GRU to a batch of facts (input fusion).

        Args:
            facts (torch.Tensor): The fact representations, shape (batch_size, max_facts_len, glove_dim).
            num_facts (List[int]): The facts lengths, shape (batch_size,).

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
        self,
        queries: torch.Tensor,
        query_lengths: torch.Tensor,
        docs: torch.Tensor,
        sentence_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """Extract the query and fact representations and apply input fusion.

        Args:
            queries (torch.Tensor): Padded queries, shape (batch_size, max_query_len).
            query_lengths (torch.Tensor): Query lenghts, shape (batch_size,).
            docs (torch.Tensor): Padded documents, shape (batch_size, max_doc_len).
            sentence_lengths (torch.Tensor): Document lenghts, shape (batch_size, max_num_sent).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, List[int]]: A tuple containing
                * the query representations, shape (batch_size, 1, rep_dim),
                * the fact representations, shape (batch_size, max_facts_len, rep_dim),
                * the number of facts, shape (batch_size,).
        """
        self.input_gru.flatten_parameters()
        self.question_gru.flatten_parameters()

        facts, num_facts = [], []
        for embedded_doc, item_sentence_lengths in zip(
            self.embedding(docs), sentence_lengths
        ):
            fact_outputs = self._get_facts(embedded_doc, item_sentence_lengths)
            facts.append(fact_outputs)
            num_facts.append(fact_outputs.shape[0])

        return (
            self._forward_queries(self.embedding(queries), query_lengths),
            self._forward_facts(facts, num_facts),
            num_facts,
        )


class AttentionGRUCell(torch.nn.Module):
    """Attention GRU cell."""

    def __init__(self, rep_dim: int, agru_dim: int) -> None:
        """Instantiate an attention GRU cell.

        Args:
            rep_dim (int): Input (fact) dimension.
            agru_dim (int): Hidden dimension.
        """
        super().__init__()
        self.Wr = torch.nn.Linear(rep_dim, agru_dim, bias=False)
        self.Ur = torch.nn.Linear(agru_dim, agru_dim)
        self.W = torch.nn.Linear(rep_dim, agru_dim, bias=False)
        self.U = torch.nn.Linear(agru_dim, agru_dim)

    def forward(
        self, fact: torch.Tensor, h_old: torch.Tensor, g_fact: torch.Tensor
    ) -> torch.Tensor:
        """A single GRU step.

        Args:
            fact (torch.Tensor): Input fact, shape (batch_size, rep_dim).
            h_old (torch.Tensor): Previous hidden states, shape (batch_size, agru_dim).
            g_fact (torch.Tensor): Attention scores for the fact, shape (batch_size, 1).

        Returns:
            torch.Tensor: The new hidden states, shape (batch_size, agru_dim).
        """
        r = torch.sigmoid(self.Wr(fact) + self.Ur(h_old))
        h_t = torch.tanh(self.W(fact) + r * self.U(h_old))

        g_fact = g_fact.expand_as(h_t)
        return g_fact * h_t + (1 - g_fact) * h_old


class AttentionGRU(torch.nn.Module):
    """Attention GRU."""

    def __init__(self, rep_dim: int, agru_dim: int) -> None:
        """Instantiate an attention GRU.

        Args:
            rep_dim (int): Input (fact) dimension.
            agru_dim (int): GRU hidden dimension.
        """
        super().__init__()
        self.agru_dim = agru_dim
        self.cell = AttentionGRUCell(rep_dim, agru_dim)

    def forward(
        self,
        facts: torch.Tensor,
        num_facts: List[int],
        g: torch.Tensor,
        mem_old: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the attention GRU to a batch.

        Args:
            facts (torch.Tensor): Input facts, shape (batch_size, max_facts_len, rep_dim).
            num_facts (List[int]): The number of facts, shape (batch_size,).
            g (torch.Tensor): Attention scores, shape (batch_size, max_facts_len, 1).
            mem_old (torch.Tensor): The previous memory, shape (batch_size, 1, rep_dim).

        Returns:
            torch.Tensor: The last hidden states, shape (batch_size, agru_dim).
        """
        _, max_num_facts, _ = facts.shape
        states = []

        # initial hidden state from previous memory
        h = mem_old.squeeze(1)
        for i in range(max_num_facts):
            fact = facts[:, i]
            h = self.cell(fact, h, g[:, i])
            states.append(h)
        states = torch.transpose(torch.stack(states), 0, 1)

        # select the correct hidden state for each example depending on the length
        return torch.stack([state[i - 1] for state, i in zip(states, num_facts)])


class MemoryModule(torch.nn.Module):
    """The episodic memory module of the DMN. This module creates interactions between queries and
    facts and applies an attention GRU.
    """

    def __init__(self, rep_dim: int, attention_dim: int, agru_dim: int) -> None:
        """Instantiate a DMN memory module.

        Args:
            rep_dim (int): The dimension of the query and fact representations and the memory.
            attention_dim (int): The dimension of the linear layer applied to the interactions.
            agru_dim (int): The hidden dimension of the attention GRU.
        """
        super().__init__()
        self.W1 = torch.nn.Linear(4 * rep_dim, attention_dim)
        self.W2 = torch.nn.Linear(attention_dim, 1)
        self.softmax = torch.nn.Softmax(dim=1)
        self.agru = AttentionGRU(rep_dim, agru_dim)
        self.Wt = torch.nn.Linear(2 * rep_dim + agru_dim, rep_dim)
        self.relu = torch.nn.ReLU()

    def _get_attention(
        self,
        facts: torch.Tensor,
        num_facts: List[int],
        queries: torch.Tensor,
        m_old: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the interactions between queries and facts and apply a linear layer.

        Args:
            facts (torch.Tensor): Input facts, shape (batch_size, max_facts_len, rep_dim).
            num_facts (List[int]): The number of facts, shape (batch_size,).
            queries (torch.Tensor): The query representations, shape (batch_size, 1, rep_dim).
            m_old (torch.Tensor): The previous memory, shape (batch_size, 1, rep_dim).

        Returns:
            torch.Tensor: Attention scores, shape (batch_size, max_facts_len, 1).
        """
        # we need to expand queries and memories to use element wise operations
        queries_expanded = queries.expand_as(facts)
        m_expanded = m_old.expand_as(facts)

        interactions = [
            facts * queries_expanded,
            facts * m_expanded,
            torch.abs(facts - queries_expanded),
            torch.abs(facts - m_expanded),
        ]
        z = torch.cat(interactions, dim=2)
        Z = self.W2(torch.tanh(self.W1(z)))

        # create mask
        batch_size, max_len, _ = facts.shape
        rng = (
            torch.arange(max_len, device=m_old.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        lengths = torch.LongTensor(num_facts).to(m_old.device)
        lengths = lengths.unsqueeze(1).expand_as(rng)
        mask = (rng < lengths).unsqueeze(-1)

        # set the values that correspond to padded tokens to -inf so they don't affect the softmax
        Z[~mask] = float("-inf")
        return self.softmax(Z)

    def forward(
        self,
        queries: torch.Tensor,
        facts: torch.Tensor,
        num_facts: List[int],
        m_old: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the next memory.

        Args:
            queries (torch.Tensor): Input queries, shape (batch_size, 1, rep_dim).
            facts (torch.Tensor): Input facts, shape (batch_size, max_facts_len, rep_dim).
            num_facts (List[int]): The number of facts, shape (batch_size,).
            m_old (torch.Tensor): The previous memory, shape (batch_size, 1, rep_dim).

        Returns:
            torch.Tensor: The next memory, shape (batch_size, 1, rep_dim).
        """
        g = self._get_attention(facts, num_facts, queries, m_old)
        c = self.agru(facts, num_facts, g, m_old)
        m = self.relu(
            self.Wt(torch.cat([m_old.squeeze(1), c, queries.squeeze(1)], dim=1))
        )
        return m.unsqueeze(1)


class AnswerModule(torch.nn.Module):
    """Answer module. Outputs relevance scores in `[0, 1]`."""

    def __init__(self, dropout: float, rep_dim: int) -> None:
        """Instantiate a DMN answer module.

        Args:
            dropout (float): Dropout value.
            rep_dim (int): The dimension of the memory.
        """
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout)
        self.linear = torch.nn.Linear(2 * rep_dim, 1)

    def forward(self, queries: torch.Tensor, memories: torch.Tensor) -> torch.Tensor:
        """Return a batch of scores.

        Args:
            queries (torch.Tensor): The query representations, shape (batch_size, rep_dim).
            memories (torch.Tensor): The final memories, shape (batch_size, rep_dim).

        Returns:
            torch.Tensor: The scores, shape (batch_size, 1).
        """
        return self.linear(self.dropout(torch.cat([queries, memories], dim=1)))


class DMNRanker(Ranker):
    """Dynamic Memory Network for ranking."""

    def __init__(
        self,
        embeddings: Vectors,
        lr: float,
        warmup_steps: int,
        hparams: Dict[str, Any],
    ) -> None:
        """Instantiate a DMN ranker.

        Args:
            embeddings (Vectors): Pre-trained embedding vectors (torchtext).
            lr (float): Learning rate.
            warmup_steps (int): Number of warmup steps.
            hparams (Dict[str, Any]): Hyperparameters.
        """
        super().__init__()
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.save_hyperparameters(hparams)

        self.input_module = InputModule(
            embeddings, hparams["rep_dim"], hparams["dropout"]
        )
        self.memory_module = MemoryModule(
            hparams["rep_dim"], hparams["attention_dim"], hparams["agru_dim"]
        )
        self.answer_module = AnswerModule(hparams["dropout"], hparams["rep_dim"])

    def forward(self, batch: DMNBatch) -> torch.Tensor:
        """Compute the relevance scores for a batch.

        Args:
            batch (DMNBatch): The queries, query lengths, docs and doc lengths.

        Returns:
            torch.Tensor: The scores, shape (batch_size, 1).
        """
        queries, facts, num_facts = self.input_module(*batch)
        m = queries
        for _ in range(self.hparams["num_episodes"]):
            m = self.memory_module(queries, facts, num_facts, m)
        return self.answer_module(queries.squeeze(1), m.squeeze(1))

    def configure_optimizers(self) -> Tuple[List[Any], List[Any]]:
        params_with_grad = filter(lambda p: p.requires_grad, self.parameters())
        opt = Adam(params_with_grad, lr=self.lr)
        sched = get_constant_schedule_with_warmup(opt, self.warmup_steps)
        return [opt], [{"scheduler": sched, "interval": "step"}]
