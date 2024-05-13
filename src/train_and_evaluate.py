from tensorflow.keras import layers
import keras_nlp as nlp 
import tensorflow.keras as keras
from sklearn.metrics import classification_report, multilabel_confusion_matrix
from sklearn.model_selection import train_test_split
from numpy import argmax
import pandas as pd
import tensorflow as tf
from importlib import reload
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.metrics import F1Score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report

def train_and_evaluate_model(create_crtnet_method, samples, one_hot_encoding_labels, classes, is_multilabel, initial_learning_rate, number_of_leads, callbacks=None, epochs=10, batch_size=64, style=None, folds=None, ):
    
    if (folds is not None):
        not_test_x, test_x, not_test_y, test_y = train_test_split(samples, one_hot_encoding_labels, test_size=0.1, random_state=42)
        kf = KFold(n_splits=folds, shuffle=True, random_state=42)
        accuracies = []
        f1_scores = []

        for train_index, val_index in kf.split(not_test_x):

            model = create_crtnet_method(
                number_of_leads=number_of_leads,
                num_classes=one_hot_encoding_labels.shape[1],
                multilabel=is_multilabel,
                learning_rate=initial_learning_rate)

            train_x, validation_x = not_test_x[train_index], not_test_x[val_index]
            train_y, validation_y = not_test_y[train_index], not_test_y[val_index]

            history = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, validation_data=(validation_x, validation_y), callbacks=callbacks, verbose=1)

            y_pred = model.predict(validation_x)

            if is_multilabel:
                y_pred = (y_pred > 0.5).astype(int)
                overall_accuracy = accuracy_score(validation_y, y_pred)
                f1 = f1_score(validation_y, y_pred, average='macro')
            else:
                validation_y = np.argmax(validation_y, axis=1)
                y_pred = np.argmax(y_pred, axis=1)
                overall_accuracy = accuracy_score(validation_y, y_pred)
                f1 = f1_score(validation_y, y_pred, average='macro')

            accuracies.append(overall_accuracy)
            f1_scores.append(f1)

        median_accuracy = np.median(accuracies)
        std_accuracy = np.std(accuracies)
        median_f1 = np.median(f1_scores)
        std_f1 = np.std(f1_scores)

        # output all the results
        print(f"Accuracy: {accuracies}")
        print(f"F1 Score: {f1_scores}")
        
        print(f"Median Accuracy: {median_accuracy:.2%}")
        print(f"Standard Deviation of Accuracy: {std_accuracy:.2%}")
        print(f"Median F1 Score: {median_f1:.2%}")
        print(f"Standard Deviation of F1 Score: {std_f1:.2%}")
    else:
        train_x, validation_x, train_y, validation_y = train_test_split(samples, one_hot_encoding_labels, test_size=0.1, random_state=42)

        model = create_crtnet_method(
                number_of_leads=number_of_leads,
                num_classes=one_hot_encoding_labels.shape[1],
                multilabel=is_multilabel,
                learning_rate=initial_learning_rate)
        
        history = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, validation_data=(validation_x, validation_y), callbacks=callbacks)

        pd.DataFrame(history.history).plot(
            figsize=(8, 5), xlim=[0, epochs], ylim=[0, 1], grid=True, xlabel="Epoch",
            style=["r--", "r--.", "b-", "b-*"] if style is None else style)
        plt.legend(loc="lower left")
        plt.show()
            
        # Classification Report
        if classes is None:
            classes = ["Class " + str(i) for i in range(len(np.unique(validation_y)))]

        y_pred = model.predict(validation_x)

        if (is_multilabel):
            # Binarize y_pred to one-hot encoding if it contains probabilities
            y_pred = (y_pred > 0.5).astype(int)

            # Calculate overall accuracy
            overall_accuracy = accuracy_score(validation_y, y_pred)
            print(f"Overall Accuracy: {overall_accuracy:.2%}")

            # calculate overall f1
            f1 = f1_score(validation_y, y_pred, average='macro')
            print(f"Overall F1 Score: {f1:.2%}")

            # Generate classification report
            report = classification_report(y_true=validation_y, y_pred=y_pred, labels=list(range(validation_y.shape[1])), target_names=classes, zero_division=0)
            print(report)

            # Accuracy per class
            accuracies_per_class = []
            for i in range(validation_y.shape[1]):
                class_accuracy = accuracy_score(validation_y[:, i], y_pred[:, i])
                accuracies_per_class.append(class_accuracy)
                print(f"Accuracy ({classes[i]}): {class_accuracy:.2%}")

        else:
            validation_y = np.argmax(validation_y, axis=1)
            y_pred = np.argmax(y_pred, axis=1)

            report = classification_report(y_true=validation_y, y_pred=y_pred, labels=np.unique(validation_y), target_names=classes,zero_division=0)
            print(report)

            # Calculate overall accuracy
            overall_accuracy = accuracy_score(validation_y, y_pred)
            print(f"Overall Accuracy: {overall_accuracy:.2%}")

            # calculate overall f1
            f1 = f1_score(validation_y, y_pred, average='macro')
            print(f"Overall F1 Score: {f1:.2%}")

            # output accuracy per class
            print('Accuracy per class:')
            for i in range(len(classes)):
                print(f'{classes[i]}: {np.round(100*sum(validation_y[validation_y==i] == y_pred[validation_y==i])/sum(validation_y==i), 2)}%')

            # Confusion Matrix
            cm = confusion_matrix(validation_y, y_pred)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt="d", cmap='Reds', xticklabels=classes, yticklabels=classes)
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.title('Confusion Matrix')
            plt.show()



physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices):
    print(f'physical devices found: {physical_devices}')
    mem_growth = tf.config.experimental.get_memory_growth(physical_devices[0])
    print(f'memory growth of dev0: {mem_growth}')
    if not mem_growth:
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
            print(f'memory growth of dev0: {tf.config.experimental.get_memory_growth(physical_devices[0])} (now enabled)')
        except:
            print(f'failed to modify device (likely already initialised)')
else:
    print('physical device not found')