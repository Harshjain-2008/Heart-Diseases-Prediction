from sklearn.metrics import accuracy_score,classification_report

def evaluate_models(models,X_test,y_test):

    results = {}

    for name, model in models.items():
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test,predictions)

        print(f"\n{name}")
        print("="*40)

        print("Accuracy: ", round(accuracy*100,2),"%")

        print(classification_report(y_test,predictions))

        results[name] = accuracy

    return results    