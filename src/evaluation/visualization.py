import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


def plot_confusion_matrix(cm):

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
    )

    disp.plot()

    plt.show()