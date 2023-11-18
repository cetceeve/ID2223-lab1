import gradio as gr
from PIL import Image
import requests
import hopsworks
import joblib
import pandas as pd

project = hopsworks.login(project="zeihers_mart")
fs = project.get_feature_store()


mr = project.get_model_registry()
model = mr.get_model("wine_model", version=1)
model_dir = model.download()
model = joblib.load(model_dir + "/wine_model.pkl")
print("Model downloaded")

def wine(alcohol, volatile_acidity, total_sulfur_dioxide, chlorides, density):
    print("Calling function")
#     df = pd.DataFrame([[sepal_length],[sepal_width],[petal_length],[petal_width]], 
    df = pd.DataFrame([[alcohol, volatile_acidity, total_sulfur_dioxide, chlorides, density]], 
                      columns=["alcohol", "volatile acidity", "total sulfur dioxide", "chlorides", "density"])
    print("Predicting")
    print(df)
    # 'res' is a list of predictions returned as the label.
    res = model.predict(df) 
    # We add '[0]' to the result of the transformed 'res', because 'res' is a list, and we only want 
    # the first element.
#     print("Res: {0}").format(res)
    print(res)
    return res[0]
        
demo = gr.Interface(
    fn=wine,
    title="Wine Predictive Analytics",
    description="Experiment with inputs to predict wine quality.",
    allow_flagging="never",
    inputs=[
        gr.Number(value=0, label="alcohol"),
        gr.Number(value=0, label="volatile acidity"),
        gr.Number(value=0, label="total sulfur dioxide"),
        gr.Number(value=0, label="chlorides"),
        gr.Number(value=0, label="density"),
        ],
    outputs=gr.Number(label="Prediction"))

demo.launch(debug=True)

