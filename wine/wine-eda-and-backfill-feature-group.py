import seaborn as sns
sns.set_palette('husl')
import matplotlib.pyplot as plt
import pandas as pd
import hopsworks


project = hopsworks.login(project="zeihers_mart")
fs = project.get_feature_store()

df = pd.read_csv("https://raw.githubusercontent.com/ID2223KTH/id2223kth.github.io/master/assignments/lab1/wine.csv")

# replace categorical variable with numerical
df["type"].replace(["white", "red"], [0, 1], inplace=True)


# alternatively, we could just drop missing data with
# df = df.dropna()

# but we fill missing data with the median of the respective value
columns_with_missing_vals = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "pH", "sulphates",
]
for col_name in columns_with_missing_vals:
    df[col_name] = df[col_name].fillna(df[col_name].median())

df.to_csv("./cleaned_dataset.csv", index=False)

print(df.info())
print(df.describe())
print(df.head())
print(df.cov())

# g = sns.heatmap(df.corr(), annot=True)
# plt.show()

# these are the features that correlate with quality
# type is not included because it is captured by the other features
df = df[["quality", "alcohol", "volatile acidity", "total sulfur dioxide", "chlorides", "density"]]
print(df.shape)

# rename for hopsworks
df = df.rename(columns={"volatile acidity": "volatile_acidity", "total sulfur dioxide": "total_sulfur_dioxide"})
# add index so hopsworks doesnt't steal data
df['index'] = df.index
wine_fg = fs.get_or_create_feature_group(
    name="wine",
    version=1,
    primary_key=["index"],
    description="Wine quality dataset")
wine_fg.insert(df)