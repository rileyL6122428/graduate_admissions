from pandas import read_csv

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
