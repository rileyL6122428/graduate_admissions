from sklearn.svm import LinearSVR

class BoundedLinearSVR(LinearSVR):

    def predict(self, X):
        predictions = super(BoundedLinearSVR, self).predict(X)
        
        bounded_predictions = []
        for prediction in predictions:
            if prediction < 0:
                bounded_predictions.append(0)
            elif prediction > 1:
                bounded_predictions.append(1)
            else:
                bounded_predictions.append(prediction)
        return bounded_predictions