from sklearn.metrics import roc_curve, auc

# Decision Tree
plt.figure(figsize=(12, 8))  # Size of the plot
plot_tree(DTmodel, filled=True, feature_names=X.columns.tolist(), class_names = ['Positive','Negative'],rounded=True)
plt.title(f"Decision Tree (max_depth={max_depth})")
plt.show()

#Make the ROC_AUC for each of the models
fpr_n,tpr_n,thresholds_n = roc_curve(y_test,y_pred_NB)
fpr_k,tpr_k,thresholds_k = roc_curve(y_test,y_pred_knn)
fpr_d,tpr_d,thresholds_d = roc_curve(y_test,y_pred_DT)
roc_auc_n = auc(fpr_n, tpr_n)
roc_auc_k = auc(fpr_k, tpr_k)
roc_auc_d = auc(fpr_d, tpr_d)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plot ROC curve for Naive Bayes
axes[0].plot(fpr_n, tpr_n, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc_n:.2f})')
axes[0].plot([0, 1], [0, 1], color='navy', linestyle='--')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curve Naive Bayes')
axes[0].legend(loc='lower right')

# Plot ROC curve for KNN
axes[1].plot(fpr_k, tpr_k, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc_k:.2f})')
axes[1].plot([0, 1], [0, 1], color='navy', linestyle='--')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('ROC Curve KNN')
axes[1].legend(loc='lower right')

# Plot ROC curve for Decision Tree
axes[2].plot(fpr_d, tpr_d, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc_d:.2f})')
axes[2].plot([0, 1], [0, 1], color='navy', linestyle='--')
axes[2].set_xlabel('False Positive Rate')
axes[2].set_ylabel('True Positive Rate')
axes[2].set_title('ROC Curve Decision Tree')
axes[2].legend(loc='lower right')

# Adjust layout to avoid overlap between subplots
plt.tight_layout()

# Show the figure with all three ROC curves
plt.show()

# Confusion matrixes
# Generate the confusion matrix
cm_NB = confusion_matrix(y_test, y_pred_NB)

# Print the confusion matrix
print("Confusion Matrix:")
print(cm_NB)

# Optional: Visualize the confusion matrix using seaborn
plt.figure(figsize=(6, 5))
sns.heatmap(cm_NB, annot=True, fmt="d", cmap="Blues", xticklabels=['Negative (0)', 'Positive (1)'], yticklabels=['Negative (0)', 'Positive (1)'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Generate the confusion matrix
cm_knn = confusion_matrix(y_test, y_pred_knn)

# Print the confusion matrix
print("Confusion Matrix:")
print(cm_knn)

# Optional: Visualize the confusion matrix using seaborn
plt.figure(figsize=(6, 5))
sns.heatmap(cm_knn, annot=True, fmt="d", cmap="Blues", xticklabels=['Negative (0)', 'Positive (1)'], yticklabels=['Negative (0)', 'Positive (1)'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Generate the confusion matrix
cm_DT = confusion_matrix(y_test, y_pred_DT)

# Print the confusion matrix
print("Confusion Matrix:")
print(cm_DT)

# Optional: Visualize the confusion matrix using seaborn
plt.figure(figsize=(6, 5))
sns.heatmap(cm_DT, annot=True, fmt="d", cmap="Blues", xticklabels=['Negative (0)', 'Positive (1)'], yticklabels=['Negative (0)', 'Positive (1)'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
