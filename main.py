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

#Question 3
dataset = pd.read_csv("heart.csv")
data = dataset.iloc[:, 0:-1]
labels = dataset.iloc[:, -1]
fig, ax = plt.subplots(ncols=4, nrows=4, figsize=(20,10))

for i in range(0, 13):
    ax.flatten()[i].hist(data.iloc[:, i])
    ax.flatten()[i].set_title(data.columns[i], fontsize=15)
    
fig.tight_layout()
plt.show()

#Question 4
dataset = pd.read_csv("heart.csv")
data = dataset.iloc[:, 0:-1]
labels = dataset.iloc[:, -1]
fig, ax = plt.subplots(ncols=4, nrows=4, figsize=(20, 10))
data.plot(ax=ax.flatten()[:13], kind='density', subplots=True, sharex=False)
fig.tight_layout()
plt.show()

#Question 5
dataset = pd.read_csv("heart.csv")
data = dataset.iloc[:, 0:-1]
labels = dataset.iloc[:, -1]
fig, ax = plt.subplots(ncols=4, nrows=4, figsize=(20, 10))

data.plot(ax=ax.flatten()[:13], kind='box', subplots=True, sharex=False, sharey=False)

fig.tight_layout()
plt.show()

#Question 6
dataset = pd.read_csv("heart.csv")
data = dataset.iloc[:, 0:-1]
labels = dataset.iloc[:, -1]
fig, ax = plt.subplots(ncols=13, nrows=13, figsize=(30, 30))

pd.plotting.scatter_matrix(data, ax=ax.flatten().reshape(13, 13))

fig.tight_layout()
plt.show()

