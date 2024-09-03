import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import List, Tuple

class RankingPredictor(nn.Module):
    def __init__(self, model_name: str = "facebook/opt-125m"):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.score_proj = nn.Linear(self.model.config.hidden_size, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_tate
        pooled_output = last_hidden_state[:, 0, :]  # Use [CLS] token representation
        score = self.score_proj(pooled_output).squeeze(-1)
        return score

    def predict_scores(self, prompts: List[str]) -> List[float]:
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048)
        with torch.no_grad():
            scores = self.forward(inputs.input_ids, inputs.attention_mask)
        return scores.tolist()

    @staticmethod
    def listMLE_loss(scores: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Compute ListMLE loss as described in the paper.
        
        Args:
        scores: Predicted scores for each prompt
        lengths: True output lengths for each prompt
        
        Returns:
        loss: ListMLE loss
        """
        sorted_lengths, indices = torch.sort(lengths, descending=True)
        sorted_scores = scores[indices]
        
        max_length = sorted_lengths.max().item()
        mask = torch.arange(max_length).expand(len(sorted_lengths), max_length) < sorted_lengths.unsqueeze(1)
        
        losses = -torch.log_softmax(sorted_scores, dim=0) * mask.float()
        loss = losses.sum() / mask.sum()
        
        return loss
