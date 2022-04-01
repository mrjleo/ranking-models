from typing import Any, Dict, Iterable, List, Tuple

import nltk
import torch
from ranking_utils.model import Ranker
from ranking_utils.model.data import DataProcessor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.optim import Adam
from torchtext.vocab import Vectors
from transformers import get_constant_schedule_with_warmup

QALSTMInput = Tuple[torch.LongTensor, torch.LongTensor]
QALSTMBatch = Tuple[
    torch.LongTensor, torch.IntTensor, torch.LongTensor, torch.IntTensor
]


class QALSTMDataProcessor(DataProcessor):
    """Data processor for QA-LSTM rankers."""

    def __init__(self, embeddings: Vectors):
        """Constructor.

        Args:
            embeddings (Vectors): Pre-trained embedding vectors (torchtext).
        """
        super().__init__()
        self.stoi = embeddings.stoi
        self.unk_id = len(self.stoi)
        self.pad_id = len(self.stoi) + 1

    def get_model_input(self, query: str, doc: str) -> QALSTMInput:
        # empty queries or documents might cause problems later on
        if len(query.strip()) == 0:
            query = "(empty)"
        if len(doc.strip()) == 0:
            doc = "(empty)"

        return (
            torch.LongTensor(
                [self.stoi.get(w, self.unk_id) for w in nltk.word_tokenize(query)]
            ),
            torch.LongTensor(
                [self.stoi.get(w, self.unk_id) for w in nltk.word_tokenize(doc)]
            ),
        )

    def get_model_batch(self, inputs: Iterable[QALSTMInput]) -> QALSTMBatch:
        batch_query_tokens, batch_doc_tokens = zip(*inputs)
        query_lengths = [len(x) for x in batch_query_tokens]
        doc_lengths = [len(x) for x in batch_doc_tokens]
        return (
            pad_sequence(
                batch_query_tokens, batch_first=True, padding_value=self.pad_id
            ),
            torch.IntTensor(query_lengths),
            pad_sequence(batch_doc_tokens, batch_first=True, padding_value=self.pad_id),
            torch.IntTensor(doc_lengths),
        )


class QALSTMRanker(Ranker):
    def __init__(
        self,
        embeddings: Vectors,
        lr: float,
        warmup_steps: int,
        hparams: Dict[str, Any],
    ):
        """Constructor.

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

        # add <unk> and <pad>
        num_embeddings = len(embeddings.vectors) + 2
        emb_dim = embeddings.vectors[0].shape[0]
        self.embedding = torch.nn.Embedding(
            num_embeddings, emb_dim, padding_idx=len(embeddings.vectors) + 1
        )

        # load pre-trained embeddings
        with torch.no_grad():
            self.embedding.weight[0 : len(embeddings.vectors)] = embeddings.vectors

        self.lstm = torch.nn.LSTM(
            emb_dim, hparams["hidden_dim"], batch_first=True, bidirectional=True
        )

        # attention weights
        self.W_am = torch.nn.Linear(
            hparams["hidden_dim"] * 2, hparams["hidden_dim"] * 2
        )
        self.W_qm = torch.nn.Linear(
            hparams["hidden_dim"] * 2, hparams["hidden_dim"] * 2
        )
        self.w_ms = torch.nn.Linear(hparams["hidden_dim"] * 2, 1)
        self.tanh = torch.nn.Tanh()
        self.softmax = torch.nn.Softmax(dim=1)

        self.dropout = torch.nn.Dropout(hparams["dropout"])
        self.cos_sim = torch.nn.CosineSimilarity()

    def _encode(self, inputs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Embed and encode a batch of padded sequences using the shared LSTM.

        Args:
            inputs (torch.Tensor): The padded input sequences.
            lengths (torch.Tensor): The sequence lengths.

        Returns:
            torch.Tensor: The LSTM outputs.
        """
        input_embed = self.embedding(inputs)
        input_seqs = pack_padded_sequence(
            input_embed, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        lstm_out, _ = self.lstm(input_seqs)
        out_seqs, _ = pad_packed_sequence(lstm_out, batch_first=True)
        return out_seqs

    def _max_pool(
        self, lstm_outputs: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        """Perform max-pooling on the LSTM outputs, masking padding tokens.

        Args:
            lstm_outputs (torch.Tensor): LSTM output sequences.
            lengths (torch.Tensor): Sequence lengths.

        Returns:
            torch.Tensor: Maximum along dimension 1.
        """
        num_sequences, max_seq_len, num_hidden = lstm_outputs.shape

        # create mask
        rng = (
            torch.arange(max_seq_len, device=lstm_outputs.device)
            .unsqueeze(0)
            .expand(num_sequences, -1)
        )
        rng = rng.unsqueeze(-1).expand(-1, -1, num_hidden)
        lengths = lengths.unsqueeze(1).expand(-1, num_hidden)
        lengths = lengths.unsqueeze(1).expand(-1, max_seq_len, -1)
        mask = rng < lengths

        # set padding outputs to -inf so they dont affect max pooling
        lstm_outputs_clone = lstm_outputs.clone()
        lstm_outputs_clone[~mask] = float("-inf")
        return torch.max(lstm_outputs_clone, dim=1)[0]

    def _attention(
        self,
        query_outputs_pooled: torch.Tensor,
        doc_outputs: torch.Tensor,
        doc_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Compute attention-weighted outputs for a batch of queries and documents, masking padding tokens.

        Args:
            query_outputs_pooled (torch.Tensor): Encoded queries after pooling.
            doc_outputs (torch.Tensor): Encoded documents.
            doc_lengths (torch.Tensor): Document lengths.

        Returns:
            torch.Tensor: Attention-weighted outputs.
        """
        # doc_outputs has shape (num_docs, max_seq_len, 2 * hidden_dim)
        # query_outputs_pooled has shape (num_docs, 2 * hidden_dim)
        # expand its shape so they match
        max_seq_len = doc_outputs.shape[1]
        m = self.tanh(
            self.W_am(doc_outputs)
            + self.W_qm(query_outputs_pooled).unsqueeze(1).expand(-1, max_seq_len, -1)
        )
        wm = self.w_ms(m)

        # mask the padding tokens before computing the softmax by setting the corresponding values to -inf
        mask = (
            torch.arange(max_seq_len, device=wm.device)[None, :] < doc_lengths[:, None]
        )
        wm[~mask] = float("-inf")

        s = self.softmax(wm)
        return doc_outputs * s

    def forward(self, batch: QALSTMBatch) -> torch.Tensor:
        """Return the similarities for all query and document pairs.

        Args:
            batch (QALSTMBatch): The input batch.

        Returns:
            torch.Tensor: The similarities.
        """
        self.lstm.flatten_parameters()
        queries, query_lengths, docs, doc_lengths = batch

        query_outputs = self._encode(queries, query_lengths)
        query_outputs_pooled = self._max_pool(query_outputs, query_lengths)

        doc_outputs = self._encode(docs, doc_lengths)
        attention = self._attention(query_outputs_pooled, doc_outputs, doc_lengths)
        attention_pooled = self._max_pool(attention, doc_lengths)

        return self.cos_sim(
            self.dropout(query_outputs_pooled), self.dropout(attention_pooled)
        ).unsqueeze(1)

    def configure_optimizers(self) -> Tuple[List[Any], List[Any]]:
        """Create an AdamW optimizer using constant schedule with warmup.

        Returns:
            Tuple[List[Any], List[Any]]: The optimizer and scheduler.
        """
        params_with_grad = filter(lambda p: p.requires_grad, self.parameters())
        opt = Adam(params_with_grad, lr=self.lr)
        sched = get_constant_schedule_with_warmup(opt, self.warmup_steps)
        return [opt], [{"scheduler": sched, "interval": "step"}]
