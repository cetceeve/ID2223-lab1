import os
import modal

LOCAL=False

if LOCAL == False:
   stub = modal.Stub("wine_daily")
   image = modal.Image.debian_slim().pip_install(["hopsworks", "scikit-learn"]) 

   @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("hopsworks-api-key"))
   def f():
       g()


def get_existing_features():
    """Returns the existing wine feature group in hopsworks as X, y dataframes"""
    import hopsworks

    project = hopsworks.login(project="zeihers_mart")
    fs = project.get_feature_store()
    wine_fg = fs.get_feature_group(name="wine", version=1)
    query = wine_fg.select_all()
    feature_view = fs.get_or_create_feature_view(
        name="wine",
        version=1,
        description="Read from Wine flower dataset",
        labels=["quality"],
        query=query
    )
    return feature_view.training_data()


def generate_wines(n):
    """Generates n new wines, with realistic value distribution and quality label"""
    import pandas as pd
    import random
    from sklearn.neighbors import KNeighborsClassifier
    import hopsworks

    alcohol = [random.uniform(8.0, 14.9) for _ in range(n)]
    density = [ # density correlates strongly with alcohol
        (((-1.0 * random.uniform(0.7, 1.3) * alcohol[i]) - (0.7*8.0)) / (1.3*14.9 - 0.7*8.0)) * (1.039-0.987) + 0.987
        for i in range(n)
    ]
    chlorides = [ # chlorides correlate weakly with alcohol
        (((-1.0 * random.uniform(0.15, 1.9) * alcohol[i]) - (0.15*8.0)) / (1.9*14.9 - 0.15*8.0)) * (0.611-0.009) + 0.009
        for i in range(n)
    ]
    df = pd.DataFrame({
        "alcohol": alcohol,
        "density": density,
        "chlorides": chlorides,
        "volatile_acidity": [random.triangular(0.08, 1.58, 0.1) for _ in range(n)],
        "total_sulfur_dioxide": [random.triangular(6.0, 440.0, 10.0) for _ in range(n)],
    })
    df = df.reindex(sorted(df.columns), axis=1)

    # assign quality based on KNN
    X, y = get_existing_features()
    X = X.reindex(sorted(X.columns), axis=1)
    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(X, y)
    df["quality"] = classifier.predict(df)

    # re-index to have unique primary keys
    max_idx = X.index.max()
    df["index"] = df.index + max_idx

    print(df.head())

    return df


def g():
    import hopsworks
    import pandas as pd

    project = hopsworks.login(project="zeihers_mart")
    fs = project.get_feature_store()

    wine_df = generate_wines(10)

    wine_fg = fs.get_feature_group(name="wine", version=1)
    wine_fg.insert(wine_df)

if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        stub.deploy("wine_daily")
        with stub.run():
            f()

