from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVR
from sklearn.metrics import r2_score
from data.as_dataframe import X_test, X_train, y_test, y_train

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


classifier = GridSearchCV(
    estimator=Pipeline([
        ('scale_features', RangeScalar([
            {
                'feature_name': 'GRE_SCORE',
                'feature_range': (0, 340)
            },
            {
                'feature_name': 'TOEFL_SCORE',
                'feature_range': (0, 120)
            },
            {
                'feature_name': 'CGPA',
                'feature_range': (0, 10)
            }
        ])),
        ('linear_svr', LinearSVR())
    ]),
    param_grid=[
        {
            'linear_svr__epsilon': [ 0 ],
            'linear_svr__C': [ 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 ],
            'linear_svr__fit_intercept': [ True ],
            'linear_svr__max_iter': [1000, 10000, 15000],
            'linear_svr__random_state': [42],
        }
    ],
    cv=5,
    scoring='r2',
)

classifier.fit(X_train, y_train)

print(classifier.best_estimator_.named_steps['linear_svr'].coef_)

print('classifier.best_params_', classifier.best_params_)
# {
#   'linear_svr__C': 14,
#   'linear_svr__epsilon': 0,
#   'linear_svr__fit_intercept': True,
#   'linear_svr__max_iter': 10000,
#   'linear_svr__random_state': 42
# }

print('classifier.best_score_', classifier.best_score_)
# 0.8065217516166696

y_pred = classifier.predict(X_test)
print('r2_score on test', r2_score(y_test, y_pred))
# 0.8186815287681184
