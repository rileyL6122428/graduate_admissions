from data.as_dataframe import X_train, y_train, X_test, y_test
from transformers.normalize import normalize
from metrics.regression_errors import regression_errors
from models.k_neighbors_iter1 import classifier

# USING X_test as neighbors alg completely couples to train set
test_predictions = classifier.predict(X_test)
errors, abs_errors = regression_errors(test_predictions, y_test)

training_copy = normalize(X_test)
training_copy['CHANCE_OF_ADMIN'] = y_test
training_copy['ERROR'] = errors
training_copy['ABS_ERROR'] = abs_errors

print('20 WORST PREDICTIONS')
print(training_copy.nlargest(20, 'ABS_ERROR'))
