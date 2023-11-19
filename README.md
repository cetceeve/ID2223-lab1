# ID2223 Lab 1
This is our report for the first lab of the scalable machine learning course.

## Task 1
For Task 1, we ran and deployed all of the components as required.
We had to make some small changes for things to be able to run, like upgrading from a deprecated version of scikit-learn.

The huggingface apps are running here:
[online prediction](https://huggingface.co/spaces/zeihers-mart/iris),
[history](https://huggingface.co/spaces/zeihers-mart/iris-monitor).

## Task 2
For task 2, we implemented all of the required components. We will go into detail about the aspectes of task 2
that required major, interesting changes compared to task 1.

You can access our UIs on huggingface to do [online prediction](https://huggingface.co/spaces/zeihers-mart/Lab1-wines)
or to see the [history](https://huggingface.co/spaces/zeihers-mart/wine-monitor) of our batch predictions of generated wines.

### Feature Engineering
The dataset was already pretty complete from the start.
All we had to do was to replace the `white` and `red` categories with numerical categories,
and to fill in some empty values. We filled emtpy values with the median value of the respective feature.

To select which features to use, we generated a correlation matrix and visualized it as a heatmap.
From here we chose the features that have strong correlations with a wine's quality.
In addition we experimented with some of the weakly correlating features which had low correlations
with the other features we had selected.
The features we selected are:
- alcohol
- density
- volatile acidity
- total sulfur dioxide
- chlorides

Interestingly, the `type` of a wine does not offer any additional information that is not
already covered by these features, since the type correlates strongly with almost all of them.

### The Model
We use an XGBoost Classifier, which is a classifier based on boosted decision trees.
We chose this model, because it deals well with a mix of category features and continuous features,
and because it tends to predict outliers well, which seems important for this task.
After all, we are interested in the really good wines.

With this model and the above features, we reach a test accuracy of `~ 63%`.

### Wine Generation
For the daily feature pipeline, we implemented a wine generation function that generates wines with similar characteristics
to the existing ones from the dataset.
The daily feature pipeline generates 10 new wines at a time.

We do this, by:
- generating random values in the correct value ranges for all features
- generating `density` and `chlorides` values based on the `alcohol` value, to ensure similar feature correlations as in the real data

With these realistic features, we use a K-Nearest-Neighbors classifier to assign
a label to the generated wine based on similar wines in the dataset.
