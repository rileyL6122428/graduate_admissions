def normalize(admission_frame):
    admission_frame_copy = admission_frame.copy()
    admission_frame_copy['GRE_SCORE'] = admission_frame_copy['GRE_SCORE'] / 340
    admission_frame_copy['TOEFL_SCORE'] = admission_frame_copy['TOEFL_SCORE'] / 120
    admission_frame_copy['CGPA'] = admission_frame_copy['CGPA'] / 10
    return admission_frame_copy
