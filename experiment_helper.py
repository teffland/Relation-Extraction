"""
Rel Classification Experiment Helper Evaluations and Diagnosis Functions
"""
import numpy as np
from scipy.stats import entropy
from matplotlib import pyplot as plt

def confusion_matrix(preds, labels, label_set, no_other=False):
    """Take predictictions, labels, and set of possible labels and calc confusion matrix

    Args:
        preds: list of predictions
        labels: list of matching ground truth labels
        label_set: list of possible labels
        other (optional): If true, calculate the micro/macro stats not considering the last label

    Returns:
        matrix: the confusion matrix with predictions along rows and truths along columns
        stats: dict of stats calculated from the confusion matrix. It contains:
            - class_precision: list of per class precisions
            - class_recall: list of per class recalls
            - class_f1: list of per class f1s
            - micro_precision: precision by summing up across all classes
            - micro_recall: recall by summing up across all classes
            - micro_f1: harmonic mean of micro_precision and micro_recall
            - macro_precision: average precision across all classes
            - macro_recall: average recall across all classes
            - macro_f1: average f1 across all classes

    Precision is defined as the predicted true positives / predicted positives
    Recall is defined as the predicted true positives / all positives
    F1 is defined as the harmonic mean of precision and recall

    Note: We input label_set separately instead of inferring it 
    because there may not be labels of every type in the precitions and labels
    """
    size = len(label_set)
    matrix = np.zeros([size, size]) # rows are predictions, columns are truths
    # fill in matrix
    for p, l in zip(preds, labels):
        matrix[p,l] += 1
    # compute class specific scores
    class_precision = np.zeros(size)
    class_recall = np.zeros(size)
    class_f1 = np.zeros(size)
    tp_sum = 0
    fp_sum = 0
    fn_sum = 0
    for label in range(size):
        tp = matrix[label, label]
        fp = np.sum(matrix[label, :]) - tp
        fn = np.sum(matrix[:, label]) - tp
        # running sums for micro (skip last if other)
        if not (no_other and label == (size -1)):
            tp_sum += tp
            fp_sum += fp
            fn_sum += fn

        # per class precision, recal, and f1
        p = tp/float(tp + fp) if tp or fp else 0
        r = tp/float(tp + fn) if tp or fn else 0
        class_precision[label] = p
        class_recall[label] = r
        class_f1[label] = 2*(p*r)/(p+r) if p or r else 0

    micro_precision = tp_sum / float(tp_sum + fp_sum) if tp_sum or fp_sum else 0
    micro_recall = tp_sum / float(tp_sum + fn_sum) if tp_sum or fn_sum else 0
    micro_f1 = (2*micro_precision*micro_recall) / (micro_precision + micro_recall)
    if no_other:
        macro_precision = np.mean(class_precision[:-1])
        macro_recall = np.mean(class_recall[:-1])
    else:
        macro_precision = np.mean(class_precision)
        macro_recall = np.mean(class_recall)
    macro_f1 = (2*macro_precision*macro_recall) / (macro_precision + macro_recall)
    stats = {'class_precision':class_precision*100,
             'class_recall':class_recall*100, 
             'class_f1':class_f1*100,
             'micro_precision':micro_precision*100, 
             'micro_recall':micro_recall*100,
             'micro_f1':micro_f1*100,
             'macro_precision':macro_precision*100, 
             'macro_recall':macro_recall*100,
             'macro_f1':macro_f1*100,
             'tp_sum':tp_sum,
             'fp_sum':fp_sum,
             'fn_sum':fn_sum}
    return matrix, stats


def directional_to_bidirectional_labels(labels, int2label):
    """Convert the directional labels to labels w/o direction (for SemEval)

    Args:
        labels: the list of label indices
        int2label: dict of label indices to label strings

    Returns:
        new_labels: `labels` with directionality removed
        new_int2label: `int2label` with directionality removed
        new_label2int: an inverse index of `new_int2label`

    Expects label names to be of the form "label(dirction)"
    Using this form (from SemEval) we split on "(" and just chop off the right side
    If the rhs doesn't exist, we assum it was unidirectional in the first place

    NOTE: Always make 'Other' last
    """
    new_labelset = list(set([label.split('(')[0] for label in int2label.values()]))
    # Always move Other to the end
    new_labelset.pop(new_labelset.index('Other'))
    new_labelset.append('Other')

    new_int2label = {i:v for i, v in enumerate(new_labelset)}
    new_label2int = {v:i for i, v in enumerate(new_labelset)}
    new_labels = [ new_label2int[int2label[label].split('(')[0]] for label in labels ]
    return new_labels, new_int2label, new_label2int

def plot_confusion_matrix(cm, label_names, save_name=None, 
                          title='Normed Confusion matrix', 
                          cmap=plt.cm.Blues, 
                          stats=None):
    """Take confusion matrix, label names and plot a very nice looking confusion matrix

    Args:
        cm: a confustion matrix w/ prediction rows and true columns
        label_names: list of class names for tick labels
        save_name (optional): if provided, save the figure to this location
        title (optional): the desired title
        cmap (optional): the colormap to display cell magnitudes with
        stats (optional): if stats, label class precisions and macro stats
    """
    fig, ax = plt.subplots(figsize=(20,20))
    
    # calc normalized cm
    x, y = np.meshgrid(range(cm.shape[0]), range(cm.shape[1]))
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    cm_normalized[np.isnan(cm_normalized)] = 0.0
    
    # print nonzero raw counts
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        norm = cm_normalized[x_val, y_val]
        c = "%i" % (cm.astype('int')[x_val, y_val])
        if norm > 0.0:
            color = 'white' if norm > .5 else 'black'
            ax.text(y_val, x_val, c, va='center', ha='center', color=color)
    
    # actual plot
    im = ax.imshow(cm_normalized, interpolation='nearest', origin='upper', cmap=cmap)
#     divider = plt.make_axes_locatable(ax)
#     cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    
    # set ticks and offset grid
    tick_marks = np.arange(len(label_names))
    tick_marks_offset = np.arange(len(label_names)) - .5
    ax.set_xticks(tick_marks, minor=False)
    ax.set_yticks(tick_marks, minor=False)
    ax.set_xticks(tick_marks_offset, minor=True)
    ax.set_yticks(tick_marks_offset, minor=True)
    ax.grid(which='minor')
    if stats:
        # include micro precisio, recall, and f1
        aug_y_labels = []
        for i in range(len(label_names)):
            aug = ("%s\nP:%0.2f, R:%0.2f, F1:%0.2f" 
                   % (label_names[i],
                      stats['class_precision'][i],
                      stats['class_recall'][i],
                      stats['class_f1'][i],))
            aug_y_labels.append(aug)
    else:
        aug_x_labels = label_names
    ax.set_xticklabels(label_names, rotation=75, horizontalalignment='left', x=1)
    ax.xaxis.tick_top()
    ax.set_yticklabels(aug_y_labels)
    
    # other stuff
    plt.tight_layout()
    plt.ylabel('Predicted Labels', fontsize=16)
    if stats:
        # include macro 
        aug_x_label = ("True Labels\n Micro P:%0.2f, R:%0.2f, F1:%0.2f\n Macro P:%0.2f, R:%0.2f, F1:%0.2f" 
                       % (stats['micro_precision'], stats['micro_recall'], stats['micro_f1'],
                          stats['macro_precision'], stats['macro_recall'], stats['macro_f1']))
    else:
        aug_x_label = "True Label"
    plt.xlabel(aug_x_label, fontsize=16)
    plt.title(title, y=1.12, fontsize=20)
    if save_name:
        plt.savefig(save_name+'.pdf')
    
def plot_dists(dists, labels, int2label):
    """Plot the predicted distributions as small multiples

    Args:
        dists: list of pmfs, all the same size
        int2label: dict of ints to labels for pmfs
    """
    if len(dists) == 1:
        num = 1
    else:
        num = 2
    fig, axarr = plt.subplots(len(dists)/num, num, sharex=True, figsize=(16, len(dists)))
    if num == 1:
        axarr = np.array(axarr, ndmin=2)

    xticks = range(len(int2label.keys()))
    for i, dist in enumerate(dists):
        pred = np.zeros_like(dist)
        pred[np.argmax(dist)] = np.max(dist)
        true = np.zeros_like(dist)
        true[labels[i]] = dist[labels[i]]
        axarr[i/2, i%2].stem(dist, 'bo-')
        axarr[i/2, i%2].stem(pred, 'ro-')
        axarr[i/2, i%2].stem(true, 'go-')

        axarr[i/2, i%2].set_xlim([-1,19])
        axarr[i/2, i%2].set_xticks(xticks)
#         if i % 3 == 0 and i:
        axarr[i/2, i%2].set_xticklabels(int2label.values(), rotation=45, horizontalalignment="right", x=-2)
        axarr[i/2, i%2].set_title("True(b): %s, entropy=%2.4f" % (int2label[labels[i]], entropy(dist)))
        axarr[i/2, i%2].set_xlabel("Predicted(g): %s" % int2label[np.argmax(dist)])
    if len(dists) > 1: 
        plt.tight_layout()

#     xticks = range(len(int2label.keys()))
#     for i, dist in enumerate(dists):
#         print(dist)
#         axarr[i/num, i%num].stem(dist)
#         axarr[i/num, i%num].stem(np.argmax(dist), color='r')
#         axarr[i/num, i%num].set_xlim([-1,19])
#         axarr[i/num, i%num].set_xticks(xticks)
# #         if i % 3 == 0 and i:
#         axarr[i/num, i%num].set_xticklabels(int2label.values(), rotation=45, horizontalalignment="right", x=-2)
#         axarr[i/num, i%num].set_title(int2label[labels[i]])
#     plt.tight_layout()
