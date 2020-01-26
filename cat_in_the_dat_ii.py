"""
@author: Advait Hasabnis
"""

import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow import keras
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

#%% AUC function


def auc(y_true, y_pred):
    def fallback_auc(y_true, y_pred):
        try:
            return roc_auc_score(y_true, y_pred)
        except:
            return 0.5
    return tf.py_function(fallback_auc, (y_true, y_pred), tf.float32)


#%% Model


def create_model(data):    
    inputs = []
    embedded_outputs = []
    
    for c in np.arange(0,len(data)):
        num_unique_values = len(np.unique(data[c]))
        embedding_dim = int(min(num_unique_values//2, 50))
        inp = keras.layers.Input(shape=(1,))
        embed = keras.layers.Embedding(num_unique_values + 1, embedding_dim)(inp)
        spdropout = keras.layers.SpatialDropout1D(0.3)(embed)
        flat = keras.layers.Flatten()(spdropout)
        inputs.append(inp)
        embedded_outputs.append(flat)
    
    concat = keras.layers.Concatenate()(embedded_outputs)
    bn1 = keras.layers.BatchNormalization()(concat)
    
    dense1 = keras.layers.Dense(300, activation="relu")(bn1)
    dropout1 = keras.layers.Dropout(0.3)(dense1)
    bn2 = keras.layers.BatchNormalization()(dropout1)
    
    dense2 = keras.layers.Dense(300, activation="relu")(bn2)
    dropout2 = keras.layers.Dropout(0.3)(dense2)
    bn3 = keras.layers.BatchNormalization()(dropout2)
    
    output = keras.layers.Dense(1, activation="sigmoid")(bn3)

    model = keras.models.Model(inputs=inputs, outputs=output)
    return model


#%% Import data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#%% Remove non-important features
train = train.drop(['bin_3'], axis=1)
test = test.drop(['bin_3'], axis=1)

#%% Handling missing data
train.loc[:, 'missing'] = train.isna().sum(axis=1)
test.loc[:, 'missing'] = test.isna().sum(axis=1)
train = train.fillna('-1').astype(str)
test = test.fillna('-1').astype(str)

#%% Replacing values that exist in test but not in train
test.loc[~test.nom_6.isin(train.nom_6.unique()), 'nom_6'] = '-1'

#%% Train and Test set
X_train = train.drop(['id', 'target'], axis=1).copy()
y_train = train.target.copy()
X_test = test.drop(['id'], axis=1).copy()

#%% Preprocessing data
preprocessor = OrdinalEncoder()

X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)
X_test = [X_test[:, k] for k in range(len(X_test.T))]

#%% Training

train_preds = np.zeros((len(train)))
test_preds = np.zeros((len(test)))

i = 1
skfolds = StratifiedKFold(n_splits=50)
for train_index, test_index in skfolds.split(X_train, y_train):
    X_train_folds, X_test_fold = X_train[train_index], X_train[test_index]
    y_train_folds, y_test_fold = y_train[train_index].values, y_train[test_index].values
    
    X_train_folds = [X_train_folds[:, k] for k in range(len(X_train_folds.T))]
    X_test_fold = [X_test_fold[:, k] for k in range(len(X_test_fold.T))]

    model = create_model(X_train_folds)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[auc])
    
    early_stopping_cb = keras.callbacks.EarlyStopping(monitor='val_auc', min_delta=0.001, patience=5,
                                                      mode='max', baseline=None, restore_best_weights=True, verbose=1)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=3, min_lr=1e-6,
                                                  mode='max', verbose=1)
    model.fit(X_train_folds,
              y_train_folds.astype('float32'),
              validation_data=(X_test_fold, y_test_fold.astype('float32')),
              verbose=1,
              batch_size=1024,
              callbacks=[early_stopping_cb, reduce_lr],
              epochs=100
              )
    
    valid_fold_preds = model.predict(X_test_fold)
    test_fold_preds = model.predict(X_test)
    train_preds[test_index] = valid_fold_preds.ravel()
    test_preds += test_fold_preds.ravel()
    print("Fold Number ",i," - AUC = ",roc_auc_score(y_test_fold.astype('float32'), valid_fold_preds))
    keras.backend.clear_session()
    i += 1
    
print("Overall AUC={}".format(roc_auc_score(train.target.values, train_preds)))

#%% Submission file

test_preds /= skfolds.get_n_splits()
dataPred = pd.DataFrame()
dataPred['id'] = test['id'].copy()
dataPred['target'] = test_preds
dataPred.to_csv('submission.csv', index=False, header=True)