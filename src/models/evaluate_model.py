# Evaluate the model
accuracy_NB = accuracy_score(y_test, y_pred_NB)
precision_NB = precision_score(y_test, y_pred_NB)
recall_NB = recall_score(y_test, y_pred_NB)
f1_NB = f1_score(y_test,y_pred_NB) 


print(f'Accuracy: {accuracy_NB}')
print(f'Precision: {precision_NB}')
print(f'Recall: {recall_NB}')
print(f'F1_score: {f1_NB}')

#Now for Knn
accuracy_knn = accuracy_score(y_test, y_pred_knn)
precision_knn = precision_score(y_test, y_pred_knn)
recall_knn = recall_score(y_test, y_pred_knn)
f1_knn = f1_score(y_test,y_pred_knn) 
print(f'Accuracy: {accuracy_knn}')
print(f'Precision: {precision_knn}')
print(f'Recall: {recall_knn}')
print(f'F1_score: {f1_knn}')

#Now for Decision Tree
accuracy_DT = accuracy_score(y_test, y_pred_DT)
precision_DT = precision_score(y_test, y_pred_DT)
recall_DT = recall_score(y_test, y_pred_DT)
f1_DT = f1_score(y_test,y_pred_DT) 
print(f'Accuracy: {accuracy_DT}')
print(f'Precision: {precision_DT}')
print(f'Recall: {recall_DT}')
print(f'F1_score: {f1_DT}')
