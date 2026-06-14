from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def train_models(X_train, y_train):
    models ={
        "Logistic Regression" : LogisticRegression(max_iter=1000),
        "Random Forest" : RandomForestClassifier(n_estimators=100,random_state=42),
        "SVM" : SVC()
    }

    for model in models.values():
        model.fit(X_train,y_train)

    return models

