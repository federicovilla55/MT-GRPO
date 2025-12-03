from ext.guardians.sentinel_metric import download_model, load_from_checkpoint
from .scorer import Scorer

class SentinelScorer(Scorer):
    def __init__(self):
        super(SentinelScorer, self).__init__()
        model_path = download_model("sapienzanlp/sentinel-ref-mqm")
        self.model = load_from_checkpoint(model_path)

    def assign_score(self, input):
        data = [{"ref": el} for el in input]
        output = self.model.predict(data, batch_size=8, gpus=1)
        return output