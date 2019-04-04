from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from data.as_dataframe import X_test, X_train, y_test, y_train
from models.range_scalar import RangeScalar
from transformers.attribute_derivation import AttributeDerivation
from transformers.attribute_picker import AttributePicker
from transformers.derive_reputation import derive_reputation

classifier = GridSearchCV(
    estimator=Pipeline([
        ('range_scalar', RangeScalar(selections=[
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
        ('derive_reputation', AttributeDerivation(
            name='REPUTATION',
            derivation=derive_reputation
        )),
        ('attribute_selection', AttributePicker(keep=[
            'GRE_SCORE',
            'TOEFL_SCORE',
            'CGPA',
            'RESEARCH',
            'REPUTATION'
        ])),
        ('k_neighbors', KNeighborsRegressor())
    ]),
    param_grid=[
        {
            'k_neighbors__n_neighbors': [ 7, 9, 11, 13, 15, 17, 19 ],
            'k_neighbors__weights': [ 'uniform', 'distance' ],
            'k_neighbors__algorithm': [ 'ball_tree', 'kd_tree' ],
            'k_neighbors__leaf_size': [ 10, 15, 20, 25, 30 ],
        }
    ],
    cv=5,
    scoring='r2'
)

classifier.fit(X_train, y_train)

print('classifier.best_params_')
print(classifier.best_params_)
# {
#   'k_neighbors__algorithm': 'ball_tree',
#   'k_neighbors__leaf_size': 10,
#   'k_neighbors__n_neighbors': 15,
#   'k_neighbors__weights': 'distance'
# }

print('classifier.best_score_')
print(classifier.best_score_)
# 0.765562012264395

y_predict = classifier.predict(X_test)
print('r2_score')
print(r2_score(y_test, y_predict))
# 0.768191456740712
