from pandas import read_csv
from sklearn.model_selection import train_test_split
admission_frame = read_csv(
    '/Users/rileylittlefield/ml-projects/admission-probs/data/Admission_Predict_Ver1.1.csv'
)

formatted_cols = [
    col_name.strip().replace(' ', '_').replace('.', '').upper()
    for col_name
    in admission_frame.columns
]

admission_frame.columns = formatted_cols

# print(admission_frame.head(10))
# print(admission_frame.columns)

X = admission_frame[[
    'GRE_SCORE',
    'TOEFL_SCORE',
    'CGPA',
    'RESEARCH',
    'LOR',
    'UNIVERSITY_RATING',
    'SOP',
]]
y = admission_frame.CHANCE_OF_ADMIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.15,
    random_state=42
)
