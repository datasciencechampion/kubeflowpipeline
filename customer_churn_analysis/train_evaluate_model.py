import global_config
import pandas as pd 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as plt
import seaborn as sns
import tensorflow as tf

from sklearn.model_selection import train_test_split

import json
import os

from tensorflow.python.lib.io import file_io 

df = pd.read_csv(global_config.processed_data_path)

#Prediction
# Creating two data frames : X => contains independent variables, y => contains dependent variable
X = df.drop(['customerID', 'Churn'], axis=1)
y = df['Churn']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

print("Train & Test Split Successfully comppleted")
'''
Feature Engineering for the Model

a. Data Conversion
    A few steps are required before we train a linear classifier with Tensorflow. we need to prepare the features to include in the model. 
    In the benchmark regression, we will use the original data without applying any transformation.

    The estimator needs to have a list of features to train the model. Hence, the column's data requires to be converted into a tensor.
    A good practice is to define two lists of features based on their type and then pass them in the feature_columns of the estimator.


Estimators use a system called feature columns to describe how the model should interpret each of the raw input features. 
An Estimator expects a vector of numeric inputs, and feature columns describe how the model should convert each feature.

Selecting and crafting the right set of feature columns is key to learning an effective model. 
A feature column can be either one of the raw inputs in the original features dict (a base feature column), or any new columns created using transformations defined over one or multiple base columns (a derived feature columns).

The linear estimator uses both numeric and categorical features. 
Feature columns work with all TensorFlow estimators and their purpose is to define the features used for modeling. Additionally, they provide some feature engineering capabilities like one-hot-encoding, normalization, and bucketization.

'''

print("Feature engineering started")

#feature extraction - numerical
tenure = tf.feature_column.numeric_column('tenure')
monthly_charges = tf.feature_column.numeric_column('MonthlyCharges')
total_charges = tf.feature_column.numeric_column('TotalCharges')

# feature extraction - categorical
col_unique_val_counts = []
cat_columns = []
for col in X.columns:
    if X[col].dtype.name != 'object':
        continue
    unique_vals = X[col].unique()
    col_unique_val_counts.append(len(unique_vals))
    cat_columns.append(col)
#    print(col, "->",unique_vals)
    

cat_cols = [tf.feature_column.categorical_column_with_hash_bucket(col, hash_bucket_size=size) 
            for col, size in zip(cat_columns, col_unique_val_counts)]

num_cols = [tenure, monthly_charges, total_charges]
feature_columns = num_cols + cat_cols

print("Feature engineering completed")
#Linear Classifier model
#We are making use of TensorFlow because we are going to use Neural Networks to classify churn..
n_classes = 2 # churn Yes or No
batch_size = 100

'''
compat.v1.estimator.inputs.pandas_input_fn
    Returns input function that would feed Pandas DataFrame into the model.In other words, return function, that has signature of ()->(dict of features, target)
        x	pandas DataFrame object.
        y	pandas Series object or DataFrame. None if absent.
        batch_size	int, size of batches to return.
        num_epochs	int, number of epochs to iterate over data. If not None, read attempts that would exceed this value will raise OutOfRangeError.
        shuffle	bool, whether to read the records in random order.
        queue_capacity	int, size of the read queue. If None, it will be set roughly to the size of x.
        num_threads	Integer, number of threads used for reading and enqueueing. In order to have predicted and repeatable order of reading and enqueueing, such as in prediction and evaluation mode, num_threads should be 1.
        target_column	str, name to give the target column y. This parameter is not used when y is a DataFrame.
'''
#We feed the model with the train set and set the number of epochs to 1000, i.e., the data will go to the model 1000 times.
input_func = tf.compat.v1.estimator.inputs.pandas_input_fn(
    x=X_train, 
    y=y_train, 
    batch_size=batch_size,
    num_epochs=1000, 
    shuffle=True)


'''
Train the Classifier
TensorFlow currently provides an estimator for the linear regression and linear classification.

Linear regression: LinearRegressor
Linear classification: LinearClassifier

The syntax of the linear classifier is the same as in the tutorial on linear regression except for one argument, n_class. We need to define the feature column, the model directory and, compare with the linear regressor; 
We have to define the number of class. For a logit regression, it the number of class is equal to 2.

The model will compute the weights of the columns contained in continuous_features and categorical_features.

Linear classifier model : tf.estimator.LinearClassifier

Train a linear model to classify instances into one of multiple possible classes. 
When number of possible classes is 2, this is binary classification.

Linear regression predicts a value while the linear classifier predicts a class. Classification aims at predicting the probability of each class given a set of inputs.

Loss is calculated by using softmax cross entropy.

ARGUMENTS:
    feature_columns	An iterable containing all the feature columns used by the model. All items in the set should be instances of classes derived from FeatureColumn.
    model_dir	Directory to save model parameters, graph and etc. This can also be used to load checkpoints from the directory into a estimator to continue training a previously saved model.
    n_classes	number of label classes. Default is binary classification. Note that class labels are integers representing the class index (i.e. values from 0 to n_classes-1). For arbitrary label values (e.g. string labels), convert to class indices first.
    weight_column	A string or a _NumericColumn created by tf.feature_column.numeric_column defining feature column representing weights. It is used to down weight or boost examples during training. It will be multiplied by the loss of the example. If it is a string, it is used as a key to fetch weight tensor from the features. If it is a _NumericColumn, raw tensor is fetched by key weight_column.key, then weight_column.normalizer_fn is applied on it to get weight tensor.
    label_vocabulary	A list of strings represents possible label values. If given, labels must be string type and have any value in label_vocabulary. If it is not given, that means labels are already encoded as integer or float within [0, 1] for n_classes=2 and encoded as integer values in {0, 1,..., n_classes-1} for n_classes>2 . Also there will be errors if vocabulary is not provided and labels are string.
    optimizer	An instance of tf.keras.optimizers.* or tf.estimator.experimental.LinearSDCA used to train the model. Can also be a string (one of 'Adagrad', 'Adam', 'Ftrl', 'RMSProp', 'SGD'), or callable. Defaults to FTRL optimizer.
    config	RunConfig object to configure the runtime settings.
    warm_start_from	A string filepath to a checkpoint to warm-start from, or a WarmStartSettings object to fully configure warm-starting. If the string filepath is provided instead of a WarmStartSettings, then all weights and biases are warm-started, and it is assumed that vocabularies and Tensor names are unchanged.
    loss_reduction	One of tf.losses.Reduction except NONE. Describes how to reduce training loss over batch. Defaults to SUM_OVER_BATCH_SIZE.
    sparse_combiner	A string specifying how to reduce if a categorical column is multivalent. One of "mean", "sqrtn", and "sum" -- these are effectively different ways to do example-level normalization, which can be useful for bag-of-words features. for more details, see tf.feature_column.linear_model.
'''

linear_model= tf.estimator.LinearClassifier(feature_columns=feature_columns, n_classes=n_classes)

'''
Trains a model given training data input_fn.

ARGUMENTS:
    input_fn	A function that provides input data for training as minibatches.
    hooks	List of tf.train.SessionRunHook subclass instances. Used for callbacks inside the training loop.
    steps	Number of steps for which to train the model. If None, train forever or train until input_fn generates the tf.errors.OutOfRange error or StopIteration exception. steps works incrementally. If you call two times train(steps=10) then training occurs in total 20 steps. If OutOfRange or StopIteration occurs in the middle, training stops before 20 steps. If you don't want to have incremental behavior please set max_steps instead. If set, max_steps must be None.
    max_steps	Number of total steps for which to train model. If None, train forever or train until input_fn generates the tf.errors.OutOfRange error or StopIteration exception. If set, steps must be None. If OutOfRange or StopIteration occurs in the middle, training stops before max_steps steps. Two calls to train(steps=100) means 200 training iterations. On the other hand, two calls to train(max_steps=100) means that the second call will not do any iteration since first call did all 100 steps.
saving_listeners	list of CheckpointSaverListener objects. Used for callbacks that run immediately before or after checkpoint savings.

RETURNS:
self, for chaining.
'''


#Let's train the model with the object model.train. 
#We use the function previously defined to feed the model with the appropriate values. 
# uinsg 10k steps. The model will be trained over a 10k steps.
print("Traning started")

linear_model.train(input_fn=input_func, steps=10000) 
#linear_model.train(input_fn=input_func, steps=100) 

print("Traning completed")
#model evaluation
#To evaluate the performance of our model, we need to use the object evaluate.
'''
Evaluates the model given evaluation data input_fn.
For each step, calls input_fn, which returns one batch of data. Evaluates until:
    =>steps batches are processed, or
    =>input_fn raises an end-of-input exception 

ARGUMENTS:
    input_fn	A function that constructs the input data for evaluation. 
    steps	Number of steps for which to evaluate model. If None, evaluates until input_fn raises an end-of-input exception.
    hooks	List of tf.train.SessionRunHook subclass instances. Used for callbacks inside the evaluation call.
    checkpoint_path	Path of a specific checkpoint to evaluate. If None, the latest checkpoint in model_dir is used. If there are no checkpoints in model_dir, evaluation is run with newly initialized Variables instead of ones restored from checkpoint.
    name	Name of the evaluation if user needs to run multiple evaluations on different data sets, such as on training data vs test data. Metrics for different evaluations are saved in separate folders, and appear separately in tensorboard.

RETURN:
A dict containing the evaluation metrics specified in model_fn keyed by name, as well as an entry global_step which contains the value of the global step for which this evaluation was performed. For canned estimators, the dict contains the loss (mean loss per mini-batch) and the average_loss (mean loss per sample). Canned classifiers also return the accuracy. Canned regressors also return the label/mean and the prediction/mean.


INFO:tensorflow:Done calling model_fn.
INFO:tensorflow:Starting evaluation at 2021-07-20T12:25:26Z
INFO:tensorflow:Graph was finalized.
INFO:tensorflow:Restoring parameters from /tmp/tmpv_raapiw/model.ckpt-10000
INFO:tensorflow:Running local_init_op.
INFO:tensorflow:Done running local_init_op.
INFO:tensorflow:Inference Time : 3.47248s
INFO:tensorflow:Finished evaluation at 2021-07-20-12:25:30
INFO:tensorflow:Saving dict for global step 10000: accuracy = 0.7662092, accuracy_baseline = 0.73166114, auc = 0.82524174, auc_precision_recall = 0.6210195, average_loss = 0.4808531, global_step = 10000, label/mean = 0.26833886, loss = 0.48399842, precision = 0.55090654, prediction/mean = 0.3707674, recall = 0.696649
INFO:tensorflow:Saving 'checkpoint_path' summary for global step 10000: /tmp/tmpv_raapiw/model.ckpt-10000
{'accuracy': 0.7662092,
 'accuracy_baseline': 0.73166114,
 'auc': 0.82524174,
 'auc_precision_recall': 0.6210195,
 'average_loss': 0.4808531,
 'label/mean': 0.26833886,
 'loss': 0.48399842,
 'precision': 0.55090654,
 'prediction/mean': 0.3707674,
 'recall': 0.696649,
 'global_step': 10000}
 
'''
#We feed the model with the test set and set the number of epochs to 1, i.e., the data will go to the model only one time.
eval_input_func = tf.compat.v1.estimator.inputs.pandas_input_fn(
      x=X_test,
      y=y_test,
      batch_size=batch_size,
      num_epochs=1,
      shuffle=False)

print("Evaluation started")
results=linear_model.evaluate(eval_input_func)
print("Evaluation completed")

model_accuracy=results['accuracy']
model_precision=results['precision']
model_loss=results['loss']
model_recall=results['recall']


# Now print to file


with tf.io.gfile.GFile(global_config.store_artifacts + "/metrics.json", 'w') as outfile:
        json.dump({ "Accuracy": str(model_accuracy), "Precision": str(model_precision), "Loss":str(model_loss), "Recall":str(model_recall)}, outfile)


print("Metrics dumped into cloud storage")


'''
Making predictions :
predict(
    input_fn, predict_keys=None, hooks=None, checkpoint_path=None,
    yield_single_examples=True
)
Returns predictions for given features.

ARGUMENTS:
    input_fn	A function that constructs the features. Prediction continues until input_fn raises an end-of-input exception (tf.errors.OutOfRangeError or StopIteration).
    predict_keys	list of str, name of the keys to predict. It is used if the tf.estimator.EstimatorSpec.predictions is a dict. If predict_keys is used then rest of the predictions will be filtered from the dictionary. If None, returns all.
    hooks	List of tf.train.SessionRunHook subclass instances. Used for callbacks inside the prediction call.
    checkpoint_path	Path of a specific checkpoint to predict. If None, the latest checkpoint in model_dir is used. If there are no checkpoints in model_dir, prediction is run with newly initialized Variables instead of ones restored from checkpoint.
    yield_single_examples	If False, yields the whole batch as returned by the model_fn instead of decomposing the batch into individual elements. This is useful if model_fn returns some tensors whose first dimension is not equal to the batch size.

RETURNS:
Evaluated values of predictions tensors.
'''

pred_input_func = tf.compat.v1.estimator.inputs.pandas_input_fn(
      x=X_test,
      batch_size=batch_size,
      num_epochs=1,
      shuffle=False)

print("Making Predictions")

preds = linear_model.predict(pred_input_func)

predictions = [p['class_ids'][0] for p in preds]

from sklearn.metrics import classification_report

target_names = ['No', 'Yes']

#print(classification_report(y_test, predictions, target_names=target_names))

'''
Model Evaluation using Confusion Matrix
  We are going to evaluate model performnace using Confusion Matrix.
  We will use:
  tf.math.confusion_matrix( 
    labels, 
    predictions, 
    num_classes=None, 
    weights=None, 
    dtype=tf.dtypes.int32,
    name=None
)
Computes the confusion matrix from predictions and labels.

ARGUMENTS:
    labels	1-D Tensor of real labels for the classification task.
    predictions	1-D Tensor of predictions for a given classification.
    num_classes	The possible number of labels the classification task can have. If this value is not provided, it will be calculated using both predictions and labels array.
    weights	An optional Tensor whose shape matches predictions.
    dtype	Data type of the confusion matrix.
    name	Scope name.

RETURNS:
A Tensor of type dtype with shape [n, n] representing the confusion matrix, where n is the number of possible labels in the classification task.

Reference : https://github.com/kubeflow/pipelines/blob/d79a835b3e433fc2439bf4ce6f4c703866f8fdcf/components/local/confusion_matrix/src/confusion_matrix.py

Model Artifacts
Create Confusion Matrix
'''

print("Evaluating Model Perfoamnce using Confusion Matrix")

confusion_metrix=tf.math.confusion_matrix(y_test,predictions).numpy()
# Printing the result
#print('Confusion_matrix: ',confusion_metrix)

unique_churn_value = list(df['Churn'].unique())
#print('Churn values :',unique_churn_value)

data = []
for target_index, target_row in enumerate(confusion_metrix):
    for predicted_index, count in enumerate(target_row):
        data.append((unique_churn_value[target_index], unique_churn_value[predicted_index], count))

df_cm = pd.DataFrame(data, columns=['target', 'predicted', 'count'])
cm_file = os.path.join(global_config.store_artifacts, 'confusion_matrix.csv')
with file_io.FileIO(cm_file, 'w') as f:
    df_cm.to_csv(f, columns=['target', 'predicted', 'count'], header=False, index=False)

    

from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test,predictions)

df_roc = pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds})
roc_file = os.path.join(global_config.store_artifacts, 'roc.csv')
with file_io.FileIO(roc_file, 'w') as f:
    df_roc.to_csv(f, columns=['fpr', 'tpr', 'thresholds'], header=False, index=False)
    
    
metadata = {
    'outputs' : [
        {
            'type': 'confusion_matrix',
            'format': 'csv',
            'schema': [
                {"name": "target", "type": "CATEGORY"},
                {"name": "predicted", "type": "CATEGORY"},
                {"name": "count", "type": "NUMBER"},
              ],
              'source': cm_file,
      # Convert unique_churn_value to string because for bealean values we want "True|False" to match csv data.
      'labels': list(map(str, unique_churn_value)),
        },
        {
            'type': 'roc',
            'format': 'csv',
            'schema': [
                {'name': 'fpr', 'type': 'NUMBER'},
                {'name': 'tpr', 'type': 'NUMBER'},
                {'name': 'thresholds', 'type': 'NUMBER'},
            ],
            'source': roc_file
        }
    
    ]
    
  }


#with file_io.FileIO('/mlpipeline-ui-metadata.json', 'w') as f:
with file_io.FileIO('/mlpipeline-ui-metadata.json', 'w') as f:
    json.dump(metadata, f)
    

    
    
print("Metadata dumped at mlpipeline-ui-metadata.json ")
metrics = {
    'metrics': [{
      'name': 'Accuracy',
      'numberValue':  str(model_accuracy),
      'format': "PERCENTAGE",
    }, 
    {
      'name': 'Precision',
      'numberValue':  str(model_precision),
      'format': "PERCENTAGE",
    }, 
    {
      'name': 'Loss',
      'numberValue':  str(model_loss),
      'format': "PERCENTAGE",
    },
    {
      'name': 'Recall',
      'numberValue':  str(model_recall),
      'format': "PERCENTAGE",
    } ,]
  }
with file_io.FileIO('/mlpipeline-metrics.json', 'w') as f:
    json.dump(metrics, f)

    
print("Metrics dumped at mlpipeline-metrics.json")
