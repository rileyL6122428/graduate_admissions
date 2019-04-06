def feature_average(featuer_names, admission_frame):
    return (
        sum([
            (admission_frame[name])
            for name
            in featuer_names
        ])
        / len(featuer_names)
    )
