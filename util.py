import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc

def draw_roc_curve(y_true, y_score):
	fpr, tpr, thresholds = roc_curve(y_true, y_score)
	roc_auc = auc(fpr, tpr)

	plt.figure()
	lw = 2
	plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic')
	plt.legend(loc="lower right")
	plt.show()

	return fpr, tpr, thresholds

def draw_confusion_matrix(y_true, y_pred):
	cm = confusion_matrix(y_true, y_pred)
	tn, fp, fn, tp = cm.ravel()
	title = 'Confusion Matrix'

	fig, ax = plt.subplots()
	im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
	ax.figure.colorbar(im, ax=ax)
	ax.set(xticks=[0, 1],
	       yticks=[0, 1],
	       xticklabels=['Negative', 'Positive'],
	       yticklabels=['Negative', 'Positive'],
	       title=title,
	       ylabel='True label',
	       xlabel='Predicted label')

	# Loop over data dimensions and create text annotations.
	thresh = cm.max() / 2.
	for i in range(cm.shape[0]):
	    for j in range(cm.shape[1]):
	        ax.text(j, i, format(cm[i, j], 'd'),
	                ha="center", va="center",
	                color="white" if cm[i, j] > thresh else "black")
	fig.tight_layout()
	plt.show()