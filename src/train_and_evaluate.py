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
import logging
import random
import numpy as np
import tensorflow as tf

# Setting seeds for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

def setup_logging(method_name):
    log_filename = datetime.datetime.now().strftime(f"output/log_{method_name}_%Y%m%d_%H%M%S.log")
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)

    file_handler = logging.FileHandler(log_filename)
    console_handler = logging.StreamHandler()

    file_handler.setLevel(logging.INFO)
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def print_and_log(logger, message):
    logger.info(message)

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

def save_and_show_plot(plt, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    plt.show()
    plt.close()

def train_model(logger, model, train_x, train_y, validation_x, validation_y, epochs, batch_size, callbacks):
    history = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, validation_data=(validation_x, validation_y), callbacks=callbacks, verbose=1)
    
    training_f1_scores = []
    validation_f1_scores = []

    for epoch in range(epochs):
        # Calculate training F1 score
        y_train_pred = model.predict(train_x)
        if len(train_y.shape) == 1 or train_y.shape[1] == 1:
            y_train_pred = np.argmax(y_train_pred, axis=1)
            train_y_epoch = np.argmax(train_y, axis=1)
        else:
            y_train_pred = (y_train_pred > 0.5).astype(int)
            train_y_epoch = train_y
        
        training_f1 = f1_score(train_y_epoch, y_train_pred, average='macro')
        training_f1_scores.append(training_f1)

        # Calculate validation F1 score
        y_val_pred = model.predict(validation_x)
        if len(validation_y.shape) == 1 or validation_y.shape[1] == 1:
            y_val_pred = np.argmax(y_val_pred, axis=1)
            validation_y_epoch = np.argmax(validation_y, axis=1)
        else:
            y_val_pred = (y_val_pred > 0.5).astype(int)
            validation_y_epoch = validation_y
        
        validation_f1 = f1_score(validation_y_epoch, y_val_pred, average='macro')
        validation_f1_scores.append(validation_f1)

    history.history['f1_score'] = training_f1_scores
    history.history['val_f1_score'] = validation_f1_scores

    # Log the metrics
    print_and_log(logger, f"Training F1 Scores: {training_f1_scores}")
    print_and_log(logger, f"Validation F1 Scores: {validation_f1_scores}")

    return history

def evaluate_model(logger, model, validation_x, validation_y, is_multilabel, classes, method_name, history, epochs, fold=None):
    y_pred = model.predict(validation_x)
    if is_multilabel:
        y_pred = (y_pred > 0.5).astype(int)
    else:
        validation_y = np.argmax(validation_y, axis=1)
        y_pred = np.argmax(y_pred, axis=1)

    accuracy = accuracy_score(validation_y, y_pred)
    f1 = f1_score(validation_y, y_pred, average='macro')
    report = classification_report(validation_y, y_pred, target_names=classes, zero_division=0)
    
    print_and_log(logger, f"Model Method: {method_name}\n{report}")
    report_filename = generate_filename('classification_report', method_name, is_multilabel, 'txt', fold)
    save_text(report_filename, report)

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
        plt.title(f'Confusion Matrix: {method_name}')
        plot_filename = generate_filename('confusion_matrix', method_name, is_multilabel, 'png', fold)
        save_and_show_plot(plt, plot_filename)
    
    # Log per class accuracies more clearly
    accuracy_logs = "\n".join([f"Class '{class_name}' Accuracy: {accuracy:.4f}" for class_name, accuracy in per_class_accuracies_dict.items()])
    print_and_log(logger, accuracy_logs)
    
    # Save classification report, confusion matrix, and per-class accuracies to file
    cm_filename = generate_filename('confusion_matrix', method_name, is_multilabel, 'json', fold)
    save_json(cm_filename, multilabel_cm.tolist() if is_multilabel else cm.tolist())
    accuracies_filename = generate_filename('per_class_accuracies', method_name, is_multilabel, 'json', fold)
    save_json(accuracies_filename, per_class_accuracies_dict)

    # Ensure all arrays in history have the same length
    min_length = min(len(v) for v in history.history.values())
    for key in history.history.keys():
        history.history[key] = history.history[key][:min_length]

    # Plot training history
    history_df = pd.DataFrame(history.history)
    history_df.plot(
        figsize=(10, 6), xlim=[0, epochs], ylim=[0, 1], grid=True, xlabel="Epoch",
        style=["r--", "r--.", "b-", "b-*", "g-.", "g-*"],
        title="Training and Validation Metrics"
    )
    plt.legend(["Train Loss", "Validation Loss", "Train Accuracy", "Validation Accuracy", "Train F1 Score", "Validation F1 Score"], loc="best")
    history_plot_filename = generate_filename('training_history_combined', method_name, is_multilabel, 'png', fold)
    save_and_show_plot(plt, history_plot_filename)

    return accuracy, f1, validation_y, y_pred

def train_and_evaluate_model(create_crtnet_method, samples, one_hot_encoding_labels, classes, is_multilabel, initial_learning_rate, number_of_leads, callbacks=None, epochs=10, batch_size=64, folds=None):

    method_name = create_crtnet_method.__name__
    logger = setup_logging(method_name)  # Initialize logging

    raw_data = {'folds': [], 'accuracies': [], 'f1_scores': []}

    if classes is None:
        classes = ["Class " + str(i) for i in range(one_hot_encoding_labels.shape[1])]

    if folds:
        not_test_x, test_x, not_test_y, test_y = train_test_split(samples, one_hot_encoding_labels, test_size=0.1, random_state=42)
        kf = KFold(n_splits=folds, shuffle=True, random_state=42)

        for fold, (train_idx, val_idx) in enumerate(kf.split(not_test_x)):
            model = create_crtnet_method(number_of_leads, one_hot_encoding_labels.shape[1], is_multilabel, initial_learning_rate)
            history = train_model(logger, model, not_test_x[train_idx], not_test_y[train_idx], not_test_x[val_idx], not_test_y[val_idx], epochs, batch_size, callbacks)

            accuracy, f1, y_true, y_pred = evaluate_model(logger, model, not_test_x[val_idx], not_test_y[val_idx], is_multilabel, classes, method_name, history, epochs, fold)

            raw_data['folds'].append({'train_idx': train_idx.tolist(), 'val_idx': val_idx.tolist(), 'y_true': y_true.tolist(), 'y_pred': y_pred.tolist()})
            raw_data['accuracies'].append(accuracy)
            raw_data['f1_scores'].append(f1)

            model_filename = generate_filename('model', method_name, is_multilabel, 'h5', fold)
            #model.save(model_filename)
            #print_and_log(logger, f"Model saved to {model_filename}")

        print_and_log(logger, f"Accuracies: {raw_data['accuracies']}")
        print_and_log(logger, f"F1 scores: {raw_data['f1_scores']}")
        print_and_log(logger, f"Median Accuracy: {np.median(raw_data['accuracies']):.2%}")
        print_and_log(logger, f"Standard Deviation of Accuracy: {np.std(raw_data['accuracies']):.2%}")
        print_and_log(logger, f"Median F1 Score: {np.median(raw_data['f1_scores']):.2%}")
        print_and_log(logger, f"Standard Deviation of F1 Score: {np.std(raw_data['f1_scores']):.2%}")

    else:
        train_x, val_x, train_y, val_y = train_test_split(samples, one_hot_encoding_labels, test_size=0.1, random_state=42)
        model = create_crtnet_method(number_of_leads, one_hot_encoding_labels.shape[1], is_multilabel, initial_learning_rate)
        history = train_model(logger, model, train_x, train_y, val_x, val_y, epochs, batch_size, callbacks)

        accuracy, f1, y_true, y_pred = evaluate_model(logger, model, val_x, val_y, is_multilabel, classes, method_name, history, epochs)

        raw_data['y_true'] = y_true.tolist()
        raw_data['y_pred'] = y_pred.tolist()
        raw_data['accuracies'].append(accuracy)
        raw_data['f1_scores'].append(f1)

        model_filename = generate_filename('model', method_name, is_multilabel, 'h5')
        #model.save(model_filename)
        #print_and_log(logger, f"Model saved to {model_filename}")

    filename = generate_filename('raw_data', method_name, is_multilabel, 'json')
    save_json(filename, raw_data)
    print_and_log(logger, f"Raw data saved to {filename}")

# GPU Configuration
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
    logger = setup_logging('gpu_setup')
    print_and_log(logger, f'Physical devices found: {physical_devices}')
    mem_growth = tf.config.experimental.get_memory_growth(physical_devices[0])
    print_and_log(logger, f'Memory growth of device 0: {mem_growth}')
    if not mem_growth:
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
            print_and_log(logger, f'Memory growth of device 0: {tf.config.experimental.get_memory_growth(physical_devices[0])} (now enabled)')
        except Exception as e:
            print_and_log(logger, f'Failed to modify device (likely already initialized): {e}')
else:
    logger = setup_logging('gpu_setup')
    print_and_log(logger, 'Physical device not found')
