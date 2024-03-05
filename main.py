import pandas as pd
import matplotlib.pyplot as plt

#Question 1
dataset = pd.read_csv("heart.csv")

#seperate data from labels
data = dataset.iloc[:, 0:-1]
labels = dataset.iloc[:, -1]

fig, ax = plt.subplots(ncols=4, nrows=4, figsize=(20, 10))

data.hist(ax=ax.flatten()[:13])

fig.tight_layout()
plt.show()
