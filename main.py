import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

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
fig, ax = plt.subplots(ncols=13, nrows=13, figsize=(20, 20))

pd.plotting.scatter_matrix(data, ax=ax)

fig.tight_layout()
plt.show()

#Question 8
df = pd.read_csv('wineQualityN.csv')

df.drop(df.columns[0], axis=1, inplace=True)

df['quality'] = df['quality'].apply(lambda x: 1 if x>= 8 else 0)
features = df.drop('quality', axis=1)
features = StandardScaler().fit_transform(features)

pca = PCA(n_components=2)
pca_result = pca.fit_transform(features)
tsne = TSNE(n_components=2, perplexity=30)
tsne_result = tsne.fit_transform(features)
plt.figure(figsize=(12, 6))

plt.subplot(1,2,1)
plt.scatter(pca_result[df['quality'] == 0, 0], pca_result[df['quality'] == 0, 1], c='pink', label='Low quality')
plt.scatter(pca_result[df['quality'] == 1, 0], pca_result[df['quality'] == 1, 1], c='red', label='High Quality')
plt.title('PCA')
plt.xlabel('Principle Compoment 1')
plt.ylabel('Principle Compoment 2') 
plt.legend()
plt.subplot(1,2,2)
plt.scatter(tsne_result[df['quality'] == 0, 0], tsne_result[df['quality'] == 0, 1], c='pink', label='Low quality')
plt.scatter(tsne_result[df['quality'] == 1, 0], tsne_result[df['quality'] == 1, 1], c='red', label='High Quality')

plt.title('t-SNE')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.legend()

plt.tight_layout()
plt.show()

#Question 9
df = pd.read_csv('wineQualityN.csv')
df.drop(df.columns[0], axis=1, inplace=True)

df['quality'] = df['quality'].apply(lambda x: 1 if x>= 8 else 0)
features = df.drop('quality', axis=1)
features = StandardScaler().fit_transform(features)

pca = PCA(n_components=11)
pca_result = pca.fit_transform(features)   

plt.figure(figsize=(8, 6))
plt.scatter(pca_result[df['quality'] == 0, 7], pca_result[df['quality'] == 0, 8], c='pink', label='Low quality')
plt.scatter(pca_result[df['quality'] == 1, 7], pca_result[df['quality'] == 1, 8], c='red', label='High Quality')
plt.title('PCA: Principal Component 8 vs Principal Component 9')
plt.xlabel('Principle Compoment 8')
plt.ylabel('Principle Compoment 9')
plt.legend()
plt.show()