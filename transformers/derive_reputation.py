def derive_reputation(admission_frame):
    return (
        admission_frame.LOR +
        admission_frame.UNIVERSITY_RATING +
        admission_frame.SOP
    ) / 15
