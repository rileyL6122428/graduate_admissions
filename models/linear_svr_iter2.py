from sklearn.svm import LinearSVR
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from models.range_scalar import RangeScalar
from models.bounded_linear_svr import BoundedLinearSVR
from transformers.attribute_derivation import AttributeDerivation
from transformers.attribute_picker import AttributePicker
from transformers.test_score_harmonic import test_score_harmonic
from transformers.derive_reputation import derive_reputation
from transformers.average_score import average_score, average_score_squared
from transformers.best_score import best_score
from transformers.feature_average import feature_average
from transformers.attribute_harmonic import feature_harmonic
from transformers.largest_feature_diff import largest_feature_diff
from data.as_dataframe import X_test, X_train, y_test, y_train
from functools import partial

# LOOK AT INSTRUCTIONS BELOW !!!!!

classifier = GridSearchCV(
    estimator=Pipeline([
        ('range_scalar', RangeScalar([
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
            },
            {
                'feature_name': 'LOR',
                'feature_range': (0, 5)
            },
            {
                'feature_name': 'UNIVERSITY_RATING',
                'feature_range': (0, 5)
            }
        ])),
        ('derive_test_score_harmonic', AttributeDerivation()),
        ('attribute_selection', AttributePicker(keep=[])),
        ('linear_svr', BoundedLinearSVR())
    ]),
    param_grid=[
        {
            'linear_svr__epsilon': [0],
            'linear_svr__C': [1, 5, 11, 14,  20, 22, 26, 30, 35, 40],
            'linear_svr__fit_intercept': [True],
            'linear_svr__max_iter': [60000],
            'linear_svr__random_state': [42],
            'derive_test_score_harmonic__derivations': [
                [
                    ('REPUTATION_HARMONIC', partial(feature_harmonic, [
                        'LOR',
                        'UNIVERSITY_RATING',
                        'SOP'
                    ])),
                    ('LARGEST_SCORE_DIFF', partial(largest_feature_diff, [
                        'GRE_SCORE',
                        'TOEFL_SCORE',
                        'CGPA'
                    ])),
                    ('AVG_SCORE', partial(feature_average, [
                        'GRE_SCORE',
                        'TOEFL_SCORE',
                        'CGPA'
                    ])),
                ]
            ],
            'attribute_selection__keep': [
                [
                    'REPUTATION_HARMONIC',
                    'AVG_SCORE',
                    'LARGEST_SCORE_DIFF'
                ]
            ]
        }
    ],
    verbose=10,
    cv=5
)

classifier.fit(X_train, y_train)

print('classifier.best_params_', classifier.best_params_)
print('classifier.coefficients', classifier.best_estimator_.named_steps['linear_svr'].coef_)

print('classifier.best_score_', classifier.best_score_)

y_pred = classifier.predict(X_test)
print('r2_score on test', r2_score(y_test, y_pred))
