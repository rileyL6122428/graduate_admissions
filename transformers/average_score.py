def average_score(admission_frame):
    return (
        admission_frame.GRE_SCORE +
        admission_frame.TOEFL_SCORE +
        admission_frame.CGPA
    ) / 3

def average_score_squared(admission_frame):
    return ((
        admission_frame.GRE_SCORE +
        admission_frame.TOEFL_SCORE +
        admission_frame.CGPA
    ) / 3) ** 2
