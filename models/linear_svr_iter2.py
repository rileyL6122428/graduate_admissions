from sklearn.svm import LinearSVR
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from models.range_scalar import RangeScalar
from transformers.attribute_derivation import AttributeDerivation
from transformers.attribute_picker import AttributePicker
from transformers.test_score_harmonic import test_score_harmonic
from transformers.derive_reputation import derive_reputation
from data.as_dataframe import X_test, X_train, y_test, y_train

derivation_picker_dicts = [
    {
        'attr_derivations': [
            ('TEST_SCORE_HARMONIC', test_score_harmonic),
            ('REPUTATION', derive_reputation)
        ],
        'attr_selections': [
            'TEST_SCORE_HARMONIC',
            'REPUTATION'
        ]
    },
    {
        'attr_derivations': [
            ('REPUTATION', derive_reputation)
        ],
        'attr_selections': [
            'GRE_SCORE',
            'TOEFL_SCORE',
            'CGPA',
            'RESEARCH',
            'REPUTATION'
        ]
    }    
]

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
            }
        ])),
        ('derive_test_score_harmonic', AttributeDerivation()),
        ('attribute_selection', AttributePicker(keep=[
            'TEST_SCORE_HARMONIC',
            'REPUTATION'
        ])),
        ('linear_svr', LinearSVR())
    ]),
    param_grid=[
        {
            'linear_svr__epsilon': [0],
            'linear_svr__C': [1, 5, 11, 14, 18],
            'linear_svr__fit_intercept': [True],
            'linear_svr__max_iter': [60000],
            'linear_svr__random_state': [42],
            'derive_test_score_harmonic__derivations': [
                dict.get('attr_derivations')
            ],
            'attribute_selection__keep': [
                dict.get('attr_selections')
            ]
        }
        for dict
        in derivation_picker_dicts
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
