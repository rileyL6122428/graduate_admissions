def best_score(admission_frame):
    return [
        max([row.GRE_SCORE, row.TOEFL_SCORE, row.CGPA])
        for _, row
        in admission_frame.iterrows()
    ]
