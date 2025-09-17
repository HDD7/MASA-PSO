import numpy as np
from sklearn.manifold import TSNE

def tsne(self, x_train, y_train, x_test, y_test):
    set = self.gbest_set

    mask = set.astype(bool)

    extended_mask = mask

    x_train_subset = x_train[:, extended_mask]
    x_test_subset = x_test[:, extended_mask]
    combined_data = np.vstack([x_train_subset, x_test_subset])
    t_sne_features = TSNE(n_components=2, learning_rate='auto', init='pca').fit_transform(combined_data)
    train_embedded = t_sne_features[:len(x_train_subset)]
    test_embedded = t_sne_features[len(x_train_subset):]
    unique_classes = np.unique(np.concatenate((y_train, y_test)))
    cmap = plt.get_cmap('RdYlGn', len(unique_classes))
    # Plot training set (circles)
    for i, cls in enumerate(unique_classes):
        mask_train = y_train == cls
        plt.scatter(train_embedded[mask_train, 0],
                    train_embedded[mask_train, 1],
                    c=[cmap(i)],
                    marker='o',
                    label=f'Train {cls}',
                    alpha=0.7,
                    edgecolors='w',
                    s=70)

    # Plot test set (triangles)
    for i, cls in enumerate(unique_classes):
        mask_test = y_test == cls
        plt.scatter(test_embedded[mask_test, 0],
                    test_embedded[mask_test, 1],
                    c=[cmap(i)],
                    marker='^',
                    label=f'Test {cls}',
                    alpha=0.7,
                    edgecolors='k',
                    s=70)
    plt.title('(b) Lung_Cancer')
    # Combine legend handles and avoid duplicates
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))  # Remove duplicates
    plt.legend(by_label.values(), by_label.keys(),
               bbox_to_anchor=(1.05, 1),
               loc='upper left',
               title='Class Groups')

    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()