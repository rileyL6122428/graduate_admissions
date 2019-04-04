class AttributeDerivation:

    def __init__(self, derivations=None):
        self.derivations = derivations

    def transform(self, X):
        copied = X.copy()
        for name, derivation in self.derivations:
            derived = derivation(X)
            copied[name] = derived
        return copied
    
    def fit(self, X, y):
        return self

    def set_params(self, **params):
        self.derivations = params.get('derivations')