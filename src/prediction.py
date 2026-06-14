def predict_patient(model,sclaer, patient_data):
    patient_sclaed = sclaer.transform([patient_data])

    prediction = model.predict(patient_sclaed)

    return prediction[0]