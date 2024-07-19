import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

names = ["Reihan", "Maulana", "Kevin", "Windi Tasya"]

def extract_features(name):
    vowels = "aeiouAEIOU"
    num_vowels = sum(1 for char in name if char in vowels)
    num_consonants = len(name) - num_vowels
    return [len(name), num_vowels, num_consonants]

data = [extract_features(name) for name in names]

X = np.array(data)

k = 2

kmeans = KMeans(n_clusters=k, random_state=0).fit(X)

labels = kmeans.labels_

fig, ax = plt.subplots()

colors = ['r', 'g']

for i in range(len(names)):
    ax.scatter(X[i][0], X[i][1], c=colors[labels[i]], label=names[i])

ax.set_xlabel('Panjang Nama')
ax.set_ylabel('Jumlah Vokal')

handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys())

plt.show()
