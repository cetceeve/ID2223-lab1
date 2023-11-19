import os
import modal
import numpy as np
    
LOCAL=False

if LOCAL == False:
   stub = modal.Stub()
   hopsworks_image = modal.Image.debian_slim().pip_install(["hopsworks","joblib","seaborn","scikit-learn","dataframe-image","xgboost"])
   @stub.function(image=hopsworks_image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("hopsworks-api-key"))
   def f():
       g()

def g():
    import pandas as pd
    import hopsworks
    import joblib
    import datetime
    from PIL import Image
    from datetime import datetime
    import dataframe_image as dfi
    from sklearn.metrics import confusion_matrix
    from matplotlib import pyplot
    import seaborn as sns

    project = hopsworks.login(project="zeihers_mart")
    fs = project.get_feature_store()
    
    mr = project.get_model_registry()
    model = mr.get_model("wine_model", version=3)
    model_dir = model.download()
    model = joblib.load(model_dir + "/wine_model.pkl")
    
    feature_view = fs.get_feature_view(name="wine", version=1)
    batch_data = feature_view.get_batch_data()
    
    y_pred = model.predict(batch_data)
    # offset for the newly generated batch
    offset = 10

    new_pred = y_pred[-offset:]
    new_pred = new_pred + 3
    # print(new_pred)
    
    # read the actual label
    wine_fg = fs.get_feature_group(name="wine", version=1)
    df = wine_fg.read() 

    now = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    data = {
        'prediction': new_pred,
        'label': df.tail(offset)["quality"],
        'datetime': np.full(offset, now),
        'index': df.tail(offset)["index"],
    }

    monitor_df = pd.DataFrame(data)
    print(monitor_df)

    # feature group for monitoring 
    monitor_fg = fs.get_or_create_feature_group(name="wine_predictions",
                                                version=1,
                                                primary_key=["index"],
                                                description="Wine quality Prediction/Outcome Monitoring"
                                                )  
    # insert the new prediction results for the generated data
    monitor_fg.insert(monitor_df, write_options={"wait_for_job" : False})
    
    history_df = monitor_fg.read()
    # Add our prediction to the history, as the history_df won't have it - 
    # the insertion was done asynchronously, so it will take ~1 min to land on App
    history_df = pd.concat([history_df, monitor_df])

    df_recent = history_df.tail(10)
    dfi.export(df_recent, './wine_df_recent.png', table_conversion = 'matplotlib')

    dataset_api = project.get_dataset_api()    
    dataset_api.upload("./wine_df_recent.png", "Resources/images", overwrite=True)
    
    predictions = history_df[['prediction']]
    labels = history_df[['label']]

    print("Number of different wine predictions to date: " + str(predictions.value_counts().count()))
    results = confusion_matrix(labels, predictions, labels=range(3, 9))
    df_cm = pd.DataFrame(results, [f"True: {s}" for s in range(3, 9)],
                            [f"Pred: {s}" for s in range(3, 9)])

    cm = sns.heatmap(df_cm, annot=True)
    fig = cm.get_figure()
    fig.savefig("./wine_confusion_matrix.png")
    dataset_api.upload("./wine_confusion_matrix.png", "Resources/images", overwrite=True)


if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        with stub.run():
            f()

