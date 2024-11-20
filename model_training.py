import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
import joblib

# Cargar los datos
df_yelp = pd.read_parquet("df_yelp.parquet")
df_atributos = pd.read_parquet("df_atributos.parquet")

# Filtrar datos relevantes
top_cities = df_yelp["city"].value_counts().nlargest(5).index
df_filtered = df_yelp[df_yelp["city"].isin(top_cities)]

# Preparar datos
preprocessor = ColumnTransformer(
    transformers=[
        ("city", OneHotEncoder(), ["city"]),
        ("num", "passthrough", ["review_count", "funny"])
    ]
)

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor())
])

X = df_filtered[["city", "review_count", "funny"]]
y = df_filtered["stars_y"]
model.fit(X, y)

# Guardar el modelo
joblib.dump(model, "model.joblib")
