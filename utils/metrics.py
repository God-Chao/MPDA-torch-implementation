

def auc(labels, probs):
    if len(probs) == 0:
        return EMPTY_DATA

    labels = _squeeze_labels(labels)
    # one hot
    labels_hot = np.eye(probs.shape[1], dtype=np.int32)[labels]
    try:
        auc_ = sklearn_metrics.roc_auc_score(labels_hot, probs)
    except ValueError as e:  # only one class in labels
        first_value = labels[0]
        for value in labels:
            if value != first_value:
                raise e
        # tf.logging.warning("when calculating auc, only one class in labels")
        auc_ = AUC_ONE_CLASS
    return auc_