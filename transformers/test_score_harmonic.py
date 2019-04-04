def test_score_harmonic(admission_frame):
    toefl_score = admission_frame['TOEFL_SCORE']
    gre_score = admission_frame['GRE_SCORE']
    cgpa = admission_frame['CGPA']
    return 3 / (1 / toefl_score + 1 / gre_score + 1 / cgpa)
