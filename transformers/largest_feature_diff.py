def largest_feature_diff(feature_names, X):
    name_combinations = [
        (name_a, name_b)
        for name_a in feature_names
        for name_b in feature_names
        if name_a != name_b
    ]

    return [
        max([
            abs(row[name_a] - row[name_b])
            for (name_a, name_b)
            in name_combinations
        ])
        for _, row
        in X.iterrows()
    ]
