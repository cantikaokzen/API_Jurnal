from imblearn.base import BaseSampler
from sklearn.neighbors import LocalOutlierFactor

class LOFResampler(BaseSampler):
    _sampling_type = "clean-sampling"
    _parameter_constraints = {}

    def __init__(self, n_neighbors=20, contamination=0.05):
        super().__init__()
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.sampling_strategy = "auto"

    def _fit_resample(self, X, y):
        lof = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            contamination=self.contamination
        )
        pred = lof.fit_predict(X)
        mask = (pred == 1)
        return X[mask], y[mask]