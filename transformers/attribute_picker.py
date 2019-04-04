class AttributePicker:

    def __init__(self, keep=None):
        self.keep = keep
    
    def transform(self, X):
        return X[self.keep]
    
    def fit(self, X, y):
        return self
    
    def set_params(self, **params):
        self.keep = params.get('keep')
