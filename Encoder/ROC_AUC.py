
import numpy as np
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import scipy
from scipy import interp
from itertools import cycle
from sklearn.metrics import roc_auc_score

def plot_ROC(y_true, y_pred, classes=[0,1,2,3]): #2
    #y = label_binarize(y, classes=classes)
    n_classes = len(classes)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    true_cl, pred_cl = list(), list()
    for i in range(n_classes):
        true_cl.append(np.where(y_true != i, 0, 1).tolist())
        pred_cl.append(np.where(y_pred != i, 0, 1).tolist())

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(true_cl[i], pred_cl[i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    true_all = list(itertools.chain.from_iterable(true_cl))
    pred_all = list(itertools.chain.from_iterable(pred_cl))

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(true_all, pred_all)
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    lw = 2
    # plt.figure()
    # plt.plot(
    #     fpr[2],
    #     tpr[2],
    #     color="darkorange",
    #     lw=lw,
    #     label="ROC curve (area = %0.2f)" % roc_auc[2],
    # )
    # plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel("False Positive Rate")
    # plt.ylabel("True Positive Rate")
    # plt.title("Receiver operating characteristic example")
    # plt.legend(loc="lower right")
    # plt.show()

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure(figsize=(7,8), dpi=100)
    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )

    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
        color="navy",
        linestyle=":",
        linewidth=4,
    )

    colors = cycle(["aqua", "darkorange", "cornflowerblue"])
    for i, color in zip(range(n_classes), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=lw,
            label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]),
        )

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.show()


from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import PrecisionRecallDisplay


def plot_PrecRec(y_true, y_pred, classes=[0,1,2,3]): #2
    #y = label_binarize(y, classes=classes)
    n_classes = len(classes)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    true_cl, pred_cl = list(), list()
    for i in range(n_classes):
        true_cl.append(np.where(y_true != i, 0, 1).tolist())
        pred_cl.append(np.where(y_pred != i, 0, 1).tolist())

    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(true_cl[i], pred_cl[i])
        average_precision[i] = average_precision_score(true_cl[i], pred_cl[i])

    true_all = list(itertools.chain.from_iterable(true_cl))
    pred_all = list(itertools.chain.from_iterable(pred_cl))

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(
        true_all, pred_all
    )
    average_precision["micro"] = average_precision_score(true_all, pred_all, average="micro")

    # setup plot details
    colors = cycle(["navy", "turquoise", "darkorange", "cornflowerblue", "teal"])

    _, ax = plt.subplots(figsize=(7, 8), dpi=100)

    f_scores = np.linspace(0.2, 0.8, num=4)
    lines, labels = [], []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        (l,) = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
        plt.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))

    display = PrecisionRecallDisplay(
        recall=recall["micro"],
        precision=precision["micro"],
        average_precision=average_precision["micro"],
    )
    display.plot(ax=ax, name="Micro-average precision-recall", color="gold")

    for i, color in zip(range(n_classes), colors):
        display = PrecisionRecallDisplay(
            recall=recall[i],
            precision=precision[i],
            average_precision=average_precision[i],
        )
        display.plot(ax=ax, name=f"Precision-recall for class {i}", color=color)

    # add the legend for the iso-f1 curves
    handles, labels = display.ax_.get_legend_handles_labels()
    handles.extend([l])
    labels.extend(["iso-f1 curves"])
    # set the legend and the axes
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(handles=handles, labels=labels, loc="best")
    ax.set_title("Precision-Recall Curve")

    plt.show()