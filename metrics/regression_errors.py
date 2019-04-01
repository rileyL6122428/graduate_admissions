def regression_errors(y_pred, y_true):
    prediction_errors = list(y_pred - y_true)
    absolute_predictions_errors = list(map(abs, prediction_errors))
    return prediction_errors, absolute_predictions_errors
    
