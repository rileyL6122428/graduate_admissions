from matplotlib import pyplot as plt

def draw_scatter(frame, attr_name, label_name, alpha, attr_bounds=None, label_bounds=None):
    plt.scatter(
        x=frame[attr_name],
        y=frame[label_name],
        alpha=alpha
    )


    plt.title('%s vs %s' % (attr_name, label_name))

    plt.xlabel(attr_name)
    if attr_bounds:
        plt.xlim(attr_bounds[0], attr_bounds[1])

    plt.ylabel(label_name)
    if label_bounds:
        plt.ylim(label_bounds[0], label_bounds[1])

    plt.show()
