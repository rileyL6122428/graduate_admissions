from data.as_dataframe import admission_frame
from matplotlib import pyplot as plt

plt.scatter(
    x=admission_frame.GRE_SCORE,
    y=admission_frame.CHANCE_OF_ADMIT,
    alpha=0.10
)

plt.title('GRE_SCORE vs CHANCE_OF_ADMIT')

plt.xlabel('GRE_SCORE')
plt.xlim(0, 340)

plt.ylabel('CHANCE_OF_ADMIT')
plt.ylim(0, 1)

plt.show()
