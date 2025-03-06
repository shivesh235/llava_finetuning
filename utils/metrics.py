from torchmetrics import Metric
from sklearn.metrics import recall_score
import torch

class VQAMetric(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
    
    def update(self, preds, target):
        self.correct += (preds == target).sum()
        self.total += len(preds)
    
    def compute(self):
        return self.correct / self.total

class OpenEndedVQAMetric(VQAMetric):
    def update(self, preds, target):
        # For open-ended questions, we might need more sophisticated matching
        # This is a simple implementation that can be extended
        self.correct += (preds == target).sum()
        self.total += len(preds)

class CloseEndedVQAMetric(VQAMetric):
    def update(self, preds, target):
        # For close-ended questions, we can do exact matching
        self.correct += (preds == target).sum()
        self.total += len(preds) 