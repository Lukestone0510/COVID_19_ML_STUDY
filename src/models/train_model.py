from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score,confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier,plot_tree

# Prepare data for scikit-learn
X = df[['Breathing Problem','Fever','Dry Cough','Sore throat','Contact with COVID Patient','Attended Large Gathering','Abroad travel']]
y = df['COVID-19']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = GaussianNB()
model.fit(X_train, y_train)

# Make predictions
y_pred_NB = model.predict(X_test)

#Now for KNN
knn_spec = KNeighborsClassifier(n_neighbors=3)
knn_fit = knn_spec.fit(X_train, y_train)
y_pred_knn = knn_fit.predict(X_test)

#Now for Decision Tree
max_depth = 7
DTmodel = DecisionTreeClassifier(max_depth=max_depth,random_state=42)
DTmodel.fit(X_train, y_train)
y_pred_DT = DTmodel.predict(X_test)
