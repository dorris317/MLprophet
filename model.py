import joblib
import pandas as pd
from prophet import Prophet
df = pd.read_csv('euro-daily-hist_1999_2022.csv')
df=df.rename(columns={r"Period\Unit:":"ds"})
df['ds']= pd.to_datetime(df["ds"])
les_colomns=df.select_dtypes(include=['object']).columns
for col in les_colomns:
    df[col]=pd.to_numeric(df[col],errors='coerce')
#on remplie les NaN avec la valeur moyenne
df=df.fillna(df.mean())
df=df.rename(columns={r"[Australian dollar ]":"y"})
df_use=df[['ds','y']]
train_size=int(len(df_use)*0.8)
X, Y=df_use[:train_size], df_use[train_size:]
model_prophet= Prophet()
model_prophet.fit(X)
joblib.dump(model_prophet,'model_prophet.joblib')