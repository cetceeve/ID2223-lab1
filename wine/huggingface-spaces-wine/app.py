import gradio as gr
from PIL import Image
import requests
import hopsworks
import joblib
import pandas as pd
import random

project = hopsworks.login(project="zeihers_mart")
fs = project.get_feature_store()


mr = project.get_model_registry()
model = mr.get_model("wine_model", version=3)
model_dir = model.download()
model = joblib.load(model_dir + "/wine_model.pkl")
print("Model downloaded")

def wine(alcohol, volatile_acidity, total_sulfur_dioxide, chlorides, density):
    print("Calling function")
    df = pd.DataFrame([[alcohol, volatile_acidity, total_sulfur_dioxide, chlorides, density]], 
                      columns=["alcohol", "volatile_acidity", "total_sulfur_dioxide", "chlorides", "density"])
    print("Predicting")
    print(df)
    res = model.predict(df) 
    print(res)
    return res[0]
        
demo = gr.Interface(
    fn=wine,
    title="Wine Predictive Analytics",
    description="Experiment with inputs to predict wine quality.",
    allow_flagging="never",
    inputs=[
        gr.Number(precision=3, value=random.uniform(8.0, 14.9), label="alcohol"),
        gr.Number(precision=3, value=random.uniform(0.08, 1.58), step=0.1, label="volatile acidity"),
        gr.Number(precision=3, value=random.uniform(6.0, 440.0), label="total sulfur dioxide"),
        gr.Number(precision=3, value=random.uniform(0.009, 0.611), step=0.01, label="chlorides"),
        gr.Number(precision=3, value=random.uniform(0.987, 1.039), step=0.01, label="density"),
    ],
    outputs=gr.Number(label="Prediction"))

demo.launch(debug=True)

