from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

tf.logging.set_verbosity(tf.logging.ERROR)

def main():
    # Data sets
    Training_Data = "test_data1.csv"
    Training_Test = "test_data2.csv"
    
    # Load datasets.
    training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
        filename=Training_Data,
        target_dtype=np.int,
        features_dtype=np.float32)
    test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
        filename=Training_Test,
        target_dtype=np.int,
        features_dtype=np.float32)

    # Specify that all features have real-value data
    feature_columns = [tf.contrib.layers.real_valued_column("", dimension=6)]
    
    # Build 3 layer DNN with 10, 20, 10 units respectively.
    classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                                hidden_units=[10, 20, 10],
                                                n_classes=2,
                                                model_dir="/tmp/test_model")
    
    # Fit model.
    classifier.fit(x=training_set.data,
                   y=training_set.target,
                   steps=10000)
    
    # Evaluate accuracy.
    accuracy_score = classifier.evaluate(x=test_set.data, y=test_set.target)["accuracy"]
    print('Accuracy: {0:f}'.format(accuracy_score))
 
main()