from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from models.range_scalar import RangeScalar
from data.as_dataframe import X_train, y_train, X_test, y_test
from transformers.attribute_derivation import AttributeDerivation
from transformers.attribute_picker import AttributePicker
from transformers.derive_reputation import derive_reputation

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
        ('derive_reputation', AttributeDerivation([
            ('REPUTATION',derive_reputation)
        ]
        )),
        ('attribute_selection', AttributePicker(keep=[
            'GRE_SCORE',
            'TOEFL_SCORE',
            'CGPA',
            'RESEARCH',
            'REPUTATION'
        ])),
        ('forest_reg', RandomForestRegressor())
    ]),
    param_grid=[
        {
            'forest_reg__n_estimators': [ 75, 100, 125 ],
            'forest_reg__criterion': [ 'mse' ],
            'forest_reg__max_depth': [ 17, 20, 23, None ],
            'forest_reg__max_features': [ 'auto', 'sqrt', 3, 4 ],
            'forest_reg__random_state': [ 42 ]
        }
    ],
    verbose=10,
    cv=5
)

classifier.fit(X_train, y_train)

print('classifier.best_params_')
print(classifier.best_params_)
# {
#   'forest_reg__criterion': 'mse',
#   'forest_reg__max_depth': 20,
#   'forest_reg__max_features': 3,
#   'forest_reg__n_estimators': 100,
#   'forest_reg__random_state': 42
# }

print('classifier.best_score_')
print(classifier.best_score_)
# 0.7876621309336417

test_predictions = classifier.predict(X_test)
print('r2_score')
print(r2_score(y_test, test_predictions))
# 0.8076722541109052
