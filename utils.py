# Calculer la matrice de confusion
from sklearn.metrics import confusion_matrix
def matrixConffudion(y_test_new,y_pred_new):
    conf_matrix = confusion_matrix(y_test_new, y_pred_new)

# Extraire les valeurs de la matrice de confusion
    num_classes = conf_matrix.shape[0]
    tn, fp, fn, tp = [], [], [], []

    if num_classes == 2:
        tn, fp, fn, tp = conf_matrix.ravel()
    else:
        for i in range(num_classes):
            for j in range(num_classes):
                if i == j:
                    tp.append(conf_matrix[i, j])
                else:
                    fn.append(conf_matrix[i, j])
                    fp.append(conf_matrix[j, i])
        tn = sum(conf_matrix.flatten()) - (sum(tp) + sum(fp) + sum(fn))

# Afficher les valeurs
    print('Confusion matrix:\n', conf_matrix)
    print('True Negative (TN):', tn)
    print('False Positive (FP):', fp)
    print('False Negative (FN):', fn)
    print('True Positive (TP):', tp)