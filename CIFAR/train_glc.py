import argparse
import os
import time
import math
import json
import numpy as np
from load_corrupted_data import CIFAR10, CIFAR100
from PIL import Image
import socket
import torchvision.transforms as transforms
from scipy.special import softmax
from tensorflow.keras import datasets, layers, models, activations
import tensorflow.keras.backend as K
import tensorflow as tf

# note: nosgdr, schedule, and epochs are highly related settings

parser = argparse.ArgumentParser(description='Trains WideResNet on CIFAR',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Optimization options
parser.add_argument('--gold_fraction', '-gf', type=float, default=0.3, help='What fraction of the data should be trusted?')
parser.add_argument('--corruption_prob', '-cprob', type=float, default=0.7, help='The label corruption probability.')
parser.add_argument('--corruption_type', '-ctype', type=str, default='unif', help='Type of corruption ("unif" or "flip").')

# random seed
parser.add_argument('--seed', type=int, default=1)
args = parser.parse_args()


mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]

train_transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor(),
     transforms.Normalize(mean, std)])
test_transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean, std)])

gold_fraction = args.gold_fraction
corruption_prob = args.corruption_prob
corruption_type = args.corruption_type
train_data_gold = CIFAR10(
        'data', True, True, gold_fraction, corruption_prob, corruption_type,
        transform=train_transform, download=True)
train_data_silver = CIFAR10(
        'data', True, False, gold_fraction, corruption_prob, corruption_type,
        transform=train_transform, download=True, shuffle_indices=train_data_gold.shuffle_indices)
train_data_gold_deterministic = CIFAR10(
        'data', True, True, gold_fraction, corruption_prob, corruption_type,
        transform=test_transform, download=True, shuffle_indices=train_data_gold.shuffle_indices)
train_all_images = np.vstack((train_data_gold.train_data, train_data_silver.train_data))
train_all_labels = np.array(train_data_gold.train_labels + train_data_silver.train_labels)
i = np.random.permutation(len(train_all_images))
train_all_images, train_all_labels = train_all_images[i], train_all_labels[i]
test_data = CIFAR10('data', train=False, transform=test_transform, download=True)
num_classes = 10

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data() # uncorrupted set

def make_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3), kernel_initializer='random_normal', bias_initializer='zeros'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='random_normal', bias_initializer='zeros'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='random_normal', bias_initializer='zeros'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu', kernel_initializer='random_normal', bias_initializer='zeros'))
    model.add(layers.Dense(10, kernel_initializer='random_normal', bias_initializer='zeros'))
    model.add(layers.Activation(activations.softmax))
    return model

# Train Silver model first 

def sparse_cat_entropy_loss(y_true, y_pred):
    y_true = K.cast(K.one_hot(K.cast(y_true[:, 0], tf.int32), num_classes), tf.float32)
    
    # scale predictions so that the class probas of each sample sum to 1
    y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    loss = y_true * K.log(y_pred)# * weights
    loss = -K.sum(loss, -1)
    return loss

model_s = make_model()
model_s.compile(optimizer='adam', loss=sparse_cat_entropy_loss,
              metrics=['sparse_categorical_accuracy'])

print('\n Silver model \n')
history_s = model_s.fit(train_data_silver.train_data, train_data_silver.train_labels, epochs=10,
                    validation_data=(test_data.test_data, test_data.test_labels))

# Form corruption matrix on silver model

probs = model_s.predict(train_data_gold_deterministic.train_data)
corruption_matrix = np.zeros((num_classes, num_classes))
label_count = np.zeros(num_classes)
for i, g_label in enumerate(train_data_gold_deterministic.train_labels):
    corruption_matrix[g_label-10] += probs[i] # -10 because added +10 to differentiate from silver labels
    label_count[g_label-10] += 1

corruption_matrix = corruption_matrix/label_count[:, np.newaxis] 

# Define GLC loss

def glc_loss(y_true, y_pred):
    silver_gold_ids = y_true[:, 0]>=10
    silver_gold_ids = tf.cast(silver_gold_ids, tf.int32)
    y_true_ = tf.where(tf.cast(silver_gold_ids, tf.bool), y_true[:, 0]-10, y_true[:, 0])
    corruption_matrix_ = tf.gather(tf.cast(corruption_matrix.T, tf.float32), tf.cast(y_true_, tf.int32))

    y_true = tf.cast(tf.one_hot(tf.cast(y_true_, tf.int32), num_classes), tf.float32)
    y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1-K.epsilon())
    probabilities = y_pred
    one_hot_labels = y_true
    
    probabilities_corrected = probabilities * corruption_matrix_
    probabilities_corrected = tf.clip_by_value(probabilities_corrected, K.epsilon(), 1-K.epsilon())

    silver_gold_ids_ = tf.expand_dims(silver_gold_ids, 1)
    mix_loss_s = -tf.reduce_sum(one_hot_labels * tf.cast(tf.math.logical_not(tf.cast(silver_gold_ids_, tf.bool)), tf.float32) * tf.log(probabilities_corrected), axis=-1)
    mix_loss_g = -tf.reduce_sum(one_hot_labels * tf.cast(silver_gold_ids_, tf.float32) * tf.log(probabilities), axis=-1)
    mix_loss = mix_loss_s + mix_loss_g
    per_example_loss =  mix_loss
    return per_example_loss

model_glc = make_model()
model_glc.compile(optimizer='adam', loss=glc_loss,
              metrics=['sparse_categorical_accuracy'])

# Define GLC model

print('\n GLC model \n')
history_glc = model_glc.fit(train_all_images, train_all_labels, epochs=10,
                    validation_data=(test_data.test_data, test_data.test_labels))

# Compare against non-GLC model

print('\n Non-GLC model \n')

model_n = make_model()
model_n.compile(optimizer='adam', loss=sparse_cat_entropy_loss,
              metrics=['sparse_categorical_accuracy'])
history_n = model_n.fit(train_all_images, train_all_labels, epochs=10,
                    validation_data=(test_data.test_data, test_data.test_labels))