from data.as_dataframe import admission_frame
from visualize_features.scatter import draw_scatter

drawings = [
    # {
    #     'attr_name': 'GRE_SCORE',
    #     'label_name': 'CHANCE_OF_ADMIT',
    #     'alpha': 0.4,
    #     'attr_bounds': None,
    #     'label_bounds': None
    # },
    # {
    #     'attr_name': 'GRE_SCORE',
    #     'label_name': 'CHANCE_OF_ADMIT',
    #     'alpha': 0.12,
    #     'attr_bounds': (0, 340),
    #     'label_bounds': (0, 1)
    # },
    {
        'attr_name': 'TOEFL_SCORE',
        'label_name': 'CHANCE_OF_ADMIT',
        'alpha': 0.25,
        'attr_bounds': None,
        'label_bounds': None
    },
    {
        'attr_name': 'TOEFL_SCORE',
        'label_name': 'CHANCE_OF_ADMIT',
        'alpha': 0.15,
        'attr_bounds': (0, 120),
        'label_bounds': (0, 1)
    },
]

for drawing in drawings:
    draw_scatter(
        admission_frame,
        drawing['attr_name'],
        drawing['label_name'],
        drawing['alpha'],
        drawing['attr_bounds'],
        drawing['label_bounds']
    )
