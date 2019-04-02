from metrics.regression_errors import regression_errors
from transformers.normalize import normalize
from data.as_dataframe import X_train, y_train
from models.forest_regressor_iter1 import classifier

training_predictions = classifier.predict(X_train)

errors, abs_erros = regression_errors(training_predictions, y_train)
training_copy = normalize(X_train)
training_copy['CHANCE_OF_ADMIN'] = y_train
training_copy['PREDICTED_CHANCE'] = training_predictions
training_copy['ERRORS'] = errors
training_copy['ABS_ERRORS'] = abs_erros

print('20 Worst')
print(training_copy.nlargest(20, 'ABS_ERRORS'))
