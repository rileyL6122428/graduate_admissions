from data.as_dataframe import admission_frame

def feature_harmonic(feature_name, admission_frame):
    return (
        len(feature_name) /
        sum([
            (1 / admission_frame[name])
            for name
            in feature_name
        ])
    )
