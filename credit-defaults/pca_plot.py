import pandas as pd
from sklearn.decomposition import RandomizedPCA
from matplotlib import pyplot as plt
from data import X, y, y_label

# Plot a 2D PCA
pca = RandomizedPCA(n_components=2)
pca_transformed_X = pca.fit_transform(X)
pca_transformed_full = pd.concat([pd.DataFrame(pca_transformed_X, columns=['Z1', 'Z2']), y], axis=1)
pca_transformed_sample = pca_transformed_full.sample(5000)
pca_transformed_sample.plot(kind='scatter', c=y_label, x='Z1', y='Z2', colormap='winter')
plt.show()
