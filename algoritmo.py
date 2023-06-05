import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('C://Users/Pichau/Desktop/Diego/NBAPlayersStats.csv')
print(df.columns)
print(df['Player'])
selected_columns = df[['Age', 'GP', 'W', 'L', 'Min', 'PTS', 'FGM', 'FGA', 'FG%', '3PM', '3PA', '3P%', 'FTM', 'FTA', 'FT%']]
print(selected_columns)
X = selected_columns.to_numpy()

cov_matrix = np.cov(X, rowvar=False)
print('---- Covariance Matrix ------')
print(cov_matrix)
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
eigenvalues1, eigenvectors1 = np.linalg.eig(cov_matrix)
print('----- Eigenvalues ------')
print(eigenvalues1)
print('----- Eigenvectors ------')
print(eigenvectors1)
print('----- Biggest Eigenvalues ------')
sorted_indices = np.argsort(eigenvalues)[::-1]
print(eigenvalues[sorted_indices[:2]])

largest_eigenvectors = eigenvectors[:, sorted_indices[:2]]

projected_data = X.dot(largest_eigenvectors)

plt.scatter(projected_data[:, 0], projected_data[:, 1], c=projected_data[:, 1], cmap='cool', alpha=0.5)
plt.xlabel('Desempenho Geral')
plt.ylabel('Consistência de Pontuação')
plt.title('PCA com os dois componentes principais')

important_columns = selected_columns.columns[sorted_indices[:2]]
plt.annotate(important_columns[0], (0.1, 0.9), xycoords='axes fraction', color='red', fontsize=12)
plt.annotate(important_columns[1], (0.1, 0.85), xycoords='axes fraction', color='red', fontsize=12)

plt.figure()

plt.subplot(2, 1, 1)
plt.hist(projected_data[:, 0], bins=20, alpha=0.5)
plt.xlabel('Desempenho Geral')
plt.ylabel('Frequencia')
plt.title('Histograma do primeiro componente')
plt.plot([], [], 'ro', label='Important Component')


plt.subplot(2, 1, 2)
plt.hist(projected_data[:, 1], bins=20, alpha=0.5)
plt.xlabel('Consistência de Pontuação')
plt.ylabel('Frequencia')
plt.title('Histograma do segundo componente')
plt.plot([], [], 'ro', label='Important Component')

plt.tight_layout()
plt.savefig('pca_histogram.jpg', dpi=300)
plt.show()
