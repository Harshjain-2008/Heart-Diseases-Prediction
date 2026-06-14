from src.data_loader import load_data
from src.preprocess import preprocess_data
from src.train import train_models
from src.evaluate import evaluate_models
from src.utils import save_model
from src.prediction import predict_patient

# load dataset 
df = load_data("data/heart.csv")

# Preprocess
(X_train,
 X_test,
 y_train,
 y_test,
 scaler 
 ) = preprocess_data(df)

# train models 
models = train_models(X_train,y_train)

# Evaluate model 
results = evaluate_models(models,X_test,y_test)

# Best model 
best_model_name = max(results, key=results.get)

best_model = models[best_model_name]

print("\n Best Model: ", best_model_name)

# save model 

save_model(best_model,"models/heart_model.pkl")
save_model(scaler,"models/scaler.pkl")

# Patient disease detector 
print("\nEnter Patient Details")

age = int(input("Age: "))
sex = int(input("Sex (0=Female, 1=Male): "))
cp = int(input("Chest Pain Type (0-3): "))
trestbps = int(input("Resting Blood Pressure: "))
chol = int(input("Cholesterol: "))
fbs = int(input("Fasting Blood Sugar (0/1): "))
restecg = int(input("Rest ECG (0-2): "))
thalach = int(input("Maximum Heart Rate: "))
exang = int(input("Exercise Induced Angina (0/1): "))
oldpeak = float(input("Oldpeak: "))
slope = int(input("Slope (0-2): "))
ca = int(input("Number of Major Vessels (0-4): "))
thal = int(input("Thal (0-3): "))

patient = [
    age,
    sex,
    cp,
    trestbps,
    chol,
    fbs,
    restecg,
    thalach,
    exang,
    oldpeak,
    slope,
    ca,
    thal
]

result = predict_patient(best_model,scaler,patient)

print("\n----Heart disease predictor----")

if result==1:
    print("Heart disease DETECTED")
else:
    print("NO disease")
