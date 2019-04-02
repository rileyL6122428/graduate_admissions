class AttributePicker:

    def __init__(self, keep):
        self.keep = keep
    
    def transform(self, X):
        return X[self.keep]
    
    def fit(self, X, y):
        return self
