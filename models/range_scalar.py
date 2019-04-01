class RangeScalar:
    def __init__(self, selections):
        self.selections = selections

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        for selection in self.selections:
            feature = X_copy[selection['feature_name']]
            feature_min, feature_max = selection['feature_range']
            transformed = (feature - feature_min) / (feature_max - feature_min)
            X_copy[selection['feature_name']] = transformed
        return X_copy
