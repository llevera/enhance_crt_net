from tensorflow.keras import layers
import keras_nlp as nlp
import tensorflow.keras as keras
from sklearn.metrics import classification_report, multilabel_confusion_matrix, accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, KFold
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import datetime
import os

def save_json(filename, data):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

def save_text(filename, data):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        f.write(data)

def generate_filename(prefix, method_name, is_multilabel, extension, fold=None):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    multilabel_str = 'multilabel' if is_multilabel else 'singlelabel'
    fold_str = f"_fold{fold}" if fold is not None else ""
    return os.path.join("output", f"{prefix}_{method_name}_{multilabel_str}_{timestamp}{fold_str}.{extension}")

def save_plot(plt, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    plt.close()

def train_model(model, train_x, train_y, validation_x, validation_y, epochs, batch_size, callbacks):
    return model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, validation_data=(validation_x, validation_y), callbacks=callbacks, verbose=1)

def evaluate_model(model, validation_x, validation_y, is_multilabel, classes, fold=None):
    y_pred = model.predict(validation_x)
    if is_multilabel:
        y_pred = (y_pred > 0.5).astype(int)
    else:
        validation_y = np.argmax(validation_y, axis=1)
        y_pred = np.argmax(y_pred, axis=1)

    accuracy = accuracy_score(validation_y, y_pred)
    f1 = f1_score(validation_y, y_pred, average='macro')
    report = classification_report(validation_y, y_pred, target_names=classes, zero_division=0)
    print(report)
    
    if is_multilabel:
        multilabel_cm = multilabel_confusion_matrix(validation_y, y_pred)
        per_class_accuracies_dict = {}
        for i, class_name in enumerate(classes):
            tn, fp, fn, tp = multilabel_cm[i].ravel()
            class_accuracy = (tp + tn) / (tp + tn + fp + fn)
            per_class_accuracies_dict[class_name] = class_accuracy
    else:
        cm = confusion_matrix(validation_y, y_pred)
        per_class_accuracies = cm.diagonal() / cm.sum(axis=1)
        per_class_accuracies_dict = {classes[i]: per_class_accuracies[i] for i in range(len(classes))}
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap='Reds', xticklabels=classes, yticklabels=classes)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.title('Confusion Matrix')
        plot_filename = generate_filename('confusion_matrix', model.name, is_multilabel, 'png', fold)
        save_plot(plt, plot_filename)
    
    print(per_class_accuracies_dict)
    
    # Save classification report, confusion matrix, and per-class accuracies to file
    report_filename = generate_filename('classification_report', model.name, is_multilabel, 'txt', fold)
    save_text(report_filename, report)
    cm_filename = generate_filename('confusion_matrix', model.name, is_multilabel, 'json', fold)
    save_json(cm_filename, multilabel_cm.tolist() if is_multilabel else cm.tolist())
    accuracies_filename = generate_filename('per_class_accuracies', model.name, is_multilabel, 'json', fold)
    save_json(accuracies_filename, per_class_accuracies_dict)

    return accuracy, f1, validation_y, y_pred

def train_and_evaluate_model(create_crtnet_method, samples, one_hot_encoding_labels, classes, is_multilabel, initial_learning_rate, number_of_leads, callbacks=None, epochs=10, batch_size=64, folds=None):

    method_name = create_crtnet_method.__name__
    raw_data = {'folds': [], 'accuracies': [], 'f1_scores': []}

    if classes is None:
        classes = ["Class " + str(i) for i in range(one_hot_encoding_labels.shape[1])]

    if folds:
        not_test_x, test_x, not_test_y, test_y = train_test_split(samples, one_hot_encoding_labels, test_size=0.1, random_state=42)
        kf = KFold(n_splits=folds, shuffle=True, random_state=42)

        for fold, (train_idx, val_idx) in enumerate(kf.split(not_test_x)):
            model = create_crtnet_method(number_of_leads, one_hot_encoding_labels.shape[1], is_multilabel, initial_learning_rate)
            history = train_model(model, not_test_x[train_idx], not_test_y[train_idx], not_test_x[val_idx], not_test_y[val_idx], epochs, batch_size, callbacks)

            # Plotting training history
            history_df = pd.DataFrame(history.history)
            history_df.plot(figsize=(8, 5), xlim=[0, epochs], ylim=[0, 1], grid=True, xlabel="Epoch")
            plt.legend(loc="lower left")
            history_plot_filename = generate_filename('training_history', method_name, is_multilabel, 'png', fold)
            save_plot(plt, history_plot_filename)

            accuracy, f1, y_true, y_pred = evaluate_model(model, not_test_x[val_idx], not_test_y[val_idx], is_multilabel, classes, fold)

            raw_data['folds'].append({'train_idx': train_idx.tolist(), 'val_idx': val_idx.tolist(), 'y_true': y_true.tolist(), 'y_pred': y_pred.tolist()})
            raw_data['accuracies'].append(accuracy)
            raw_data['f1_scores'].append(f1)

            model_filename = generate_filename('model', method_name, is_multilabel, 'h5', fold)
            #model.save(model_filename)
            #print(f"Model saved to {model_filename}")

        print(f"Accuracies: {raw_data['accuracies']}")
        print(f"F1 scores: {raw_data['f1_scores']}")
        print(f"Median Accuracy: {np.median(raw_data['accuracies']):.2%}")
        print(f"Standard Deviation of Accuracy: {np.std(raw_data['accuracies']):.2%}")
        print(f"Median F1 Score: {np.median(raw_data['f1_scores']):.2%}")
        print(f"Standard Deviation of F1 Score: {np.std(raw_data['f1_scores']):.2%}")

    else:
        train_x, val_x, train_y, val_y = train_test_split(samples, one_hot_encoding_labels, test_size=0.1, random_state=42)
        model = create_crtnet_method(number_of_leads, one_hot_encoding_labels.shape[1], is_multilabel, initial_learning_rate)
        history = train_model(model, train_x, train_y, val_x, val_y, epochs, batch_size, callbacks)

        # Plotting training history
        history_df = pd.DataFrame(history.history)
        history_df.plot(figsize=(8, 5), xlim=[0, epochs], ylim=[0, 1], grid=True, xlabel="Epoch")
        plt.legend(loc="lower left")
        history_plot_filename = generate_filename('training_history', method_name, is_multilabel, 'png')
        save_plot(plt, history_plot_filename)

        accuracy, f1, y_true, y_pred = evaluate_model(model, val_x, val_y, is_multilabel, classes)

        raw_data['y_true'] = y_true.tolist()
        raw_data['y_pred'] = y_pred.tolist()
        raw_data['accuracies'].append(accuracy)
        raw_data['f1_scores'].append(f1)

        model_filename = generate_filename('model', method_name, is_multilabel, 'h5')
        #model.save(model_filename)
        #print(f"Model saved to {model_filename}")

    filename = generate_filename('raw_data', method_name, is_multilabel, 'json')
    save_json(filename, raw_data)
    print(f"Raw data saved to {filename}")

# GPU Configuration
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
    print(f'Physical devices found: {physical_devices}')
    mem_growth = tf.config.experimental.get_memory_growth(physical_devices[0])
    print(f'Memory growth of device 0: {mem_growth}')
    if not mem_growth:
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
            print(f'Memory growth of device 0: {tf.config.experimental.get_memory_growth(physical_devices[0])} (now enabled)')
        except Exception as e:
            print(f'Failed to modify device (likely already initialized): {e}')
else:
    print('Physical device not found')
