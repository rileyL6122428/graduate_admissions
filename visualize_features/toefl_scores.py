from data.as_dataframe import admission_frame
from matplotlib import pyplot as plt


plt.scatter(
    x=admission_frame.TOEFL_SCORE,
    y=admission_frame.CHANCE_OF_ADMIT,
    alpha=0.5
)

plt.title('TOEFL_SCORE vs CHANCE_OF_ADMIT')

plt.xlabel('TOEFL_SCORE')
plt.ylabel('CHANCE_OF_ADMIT')

plt.show()