class AttributeDerivation:

    def __init__(self, name, derivation):
        self.name = name
        self.derivation = derivation

    def transform(self, X):
        derived = self.derivation(X)
        copied = X.copy()
        copied[self.name] = derived
        return copied
    
    def fit(self, X, y):
        return self