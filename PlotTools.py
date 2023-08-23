import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix(cm, labels_x, labels_y=None, figsize=None):
    """
    Nice plot of confusion matrix. Diagonal uses green colorscale, off-diagonal uses red colorscale.
    If only labels_x is provided these will be used for y as well.
    :param cm: confusion matrix
    :type cm: np.ndarray
    :param labels_x: Labels/categories for x-axis
    :type labels_x: list
    :param labels_y: Optional labels/categories for y-axis
    :type labels_y: list
    :param figsize: Optional figure size. Default is [5, 5].
    :type figsize: list
    :return: confusion matrix figure
    :rtype: matplotlib.figure.Figure
    """
    # Support for total columns not done yet
    labels_y = labels_y or labels_x
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    diag_mask = np.eye(cm.shape[0], dtype=bool)

    anno_labels = [[str(j) for j in range(cm.shape[1])] for i in range(cm.shape[0])]
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            anno_labels[i][j] = f"{cm[i,j]}\n{cm_norm[i,j]:.0%}" if cm[i,j] else ''

    fig = plt.figure(figsize=figsize or [5, 5])
    sns.heatmap(cm_norm, annot=anno_labels, fmt='', mask=~diag_mask, cmap='Greens', vmin=0, vmax=1.4, cbar=False, linewidths=2)
    sns.heatmap(cm_norm, annot=anno_labels, fmt='', mask=diag_mask, cmap='Oranges', vmin=0, vmax=1.2, cbar=False, linewidths=2,
                xticklabels=labels_x, yticklabels=labels_y)

    ax = plt.gca()
    ax.set_title('Confusion matrix', fontsize=14)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    plt.show()
    return fig


if __name__ == '__main__':
    cm = np.array([[1, 2],
                   [3, 4]])

    labels_x = ['Accept', 'Reject']
    labels_y = ['Good', 'Bad']

    plot_confusion_matrix(cm, labels_x, labels_y)

    _ = 'bp'