from fastapi import FastAPI, UploadFile
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
#import uvicorn

app = FastAPI()
model_prophet = Prophet()
#def upload_csv(file: UploadFile):
    

    #return df_use

@app.post("/predict/")
exitdef predict(file: UploadFile):
    df = pd.read_csv(file.file)
    df = df.rename(columns={r"Period\Unit:": "ds"})
    df["ds"] = pd.to_datetime(df["ds"])

    # Conversion des colonnes numériques
    les_colonnes = df.select_dtypes(include=["object"]).columns
    for col in les_colonnes:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remplissage des valeurs manquantes avec la moyenne
    df = df.fillna(df.mean())
    df = df.rename(columns={r"[Australian dollar ]": "y"})
    df_use = df[["ds", "y"]]
    
    train_size = int(len(df_use) * 0.8)
    X, Y = df_use[:train_size], df_use[train_size:]

    model_prophet.fit(X)
    future = model_prophet.make_future_dataframe(periods=365)
    forecast = model_prophet.predict(future)

    predictions = forecast[["ds", "yhat"]].to_dict(orient="records")

    return {"predictions": predictions}
# if __name__ == "__main__":
#     uvicorn.run(app, host="127.0.0.1", port=8443)