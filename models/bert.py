from typing import Any, Dict, Iterable, List, Tuple

import torch
import transformers
from ranking_utils.model import Ranker
from ranking_utils.model.data import DataProcessor
from transformers import BertModel, BertTokenizer, get_constant_schedule_with_warmup

BertInput = Tuple[str, str]
BertBatch = Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]


class BertDataProcessor(DataProcessor):
    """Data processor for cross-attention BERT rankers."""

    def __init__(self, bert_model: str, char_limit: int):
        """Constructor.

        Args:
            bert_model (str, optional): Pre-trained BERT model.
            char_limit (int, optional): Maximum number of characters per query/document.
        """
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)
        self.char_limit = char_limit

        # without this, there will be a message for each tokenizer call
        transformers.logging.set_verbosity_error()

    def get_model_input(self, query: str, doc: str) -> BertInput:
        # empty queries or documents might cause problems later on
        if len(query.strip()) == 0:
            query = "(empty)"
        if len(doc.strip()) == 0:
            doc = "(empty)"

        # limit characters to avoid tokenization bottlenecks
        return query[: self.char_limit], doc[: self.char_limit]

    def get_model_batch(self, inputs: Iterable[BertInput]) -> BertBatch:
        queries, docs = zip(*inputs)
        inputs = self.tokenizer(queries, docs, padding=True, truncation=True)
        return (
            torch.LongTensor(inputs["input_ids"]),
            torch.LongTensor(inputs["attention_mask"]),
            torch.LongTensor(inputs["token_type_ids"]),
        )


class BertRanker(Ranker):
    """Cross-attention BERT ranker."""

    def __init__(
        self, lr: float, warmup_steps: int, hparams: Dict[str, Any],
    ):
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

        self.bert = BertModel.from_pretrained(hparams["bert_model"], return_dict=True)
        self.dropout = torch.nn.Dropout(hparams["dropout"])
        self.classification = torch.nn.Linear(
            self.bert.encoder.layer[-1].output.dense.out_features, 1
        )

        for p in self.bert.parameters():
            p.requires_grad = not hparams["freeze_bert"]

    def forward(self, batch: BertBatch) -> torch.Tensor:
        """Compute the relevance scores for a batch.

        Args:
            batch (BertBatch): BERT inputs.

        Returns:
            torch.Tensor: The output scores, shape (batch_size, 1).
        """
        cls_out = self.bert(*batch)["last_hidden_state"][:, 0]
        return self.classification(self.dropout(cls_out))

    def configure_optimizers(self) -> Tuple[List[Any], List[Any]]:
        """Create an AdamW optimizer using constant schedule with warmup.

        Returns:
            Tuple[List[Any], List[Any]]: The optimizer and scheduler.
        """
        opt = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr
        )
        sched = get_constant_schedule_with_warmup(opt, self.warmup_steps)
        return [opt], [{"scheduler": sched, "interval": "step"}]
