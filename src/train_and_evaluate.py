from tensorflow.keras import layers
import keras_nlp as nlp
import tensorflow.keras as keras
from sklearn.metrics import classification_report, multilabel_confusion_matrix, accuracy_score, f1_score, confusion_matrix
from sklearn.metrics import hamming_loss, accuracy_score, f1_score, average_precision_score
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
    # Train the model
    history = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, validation_data=(validation_x, validation_y), callbacks=callbacks, verbose=1)

    # Log the completion of training
    print_and_log(logger, "Training completed")

    return history

def evaluate_model(logger, model, validation_x, validation_y, is_multilabel, classes, method_name, history, epochs, fold=None):
    y_pred = model.predict(validation_x, verbose=1)
    eval_results = model.evaluate(validation_x, validation_y, verbose=1)
    if is_multilabel:
        y_pred = (y_pred > 0.5).astype(int)
    else:
        validation_y = np.argmax(validation_y, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
    
    y_pred = model.predict(validation_x)
    if is_multilabel:
        y_pred = (y_pred > 0.5).astype(int)
    else:
        validation_y = np.argmax(validation_y, axis=1)
        y_pred = np.argmax(y_pred, axis=1)

    report = classification_report(validation_y, y_pred, target_names=classes, zero_division=0)
    
    # Log and save classification report
    print_and_log(logger, f"Model Method: {method_name}\n{report}")

    # Confusion matrix
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
    
    # Log per class accuracies
    accuracy_logs = "\n".join([f"Class '{class_name}' Accuracy: {accuracy:.4f}" for class_name, accuracy in per_class_accuracies_dict.items()])
    print_and_log(logger, accuracy_logs)
    
    # Save confusion matrix
    cm_filename = generate_filename('confusion_matrix', method_name, is_multilabel, 'json', fold)

    # Plot training history
    history_df = pd.DataFrame(history.history)
    history_df.plot(
        figsize=(10, 6), xlim=[0, epochs], ylim=[0, 1], grid=True, xlabel="Epoch",
        title="Training and Validation Metrics"
    )
    plt.ylabel('Metric Value')
    history_plot_filename = generate_filename('training_history_combined', method_name, is_multilabel, 'png', fold)
    save_and_show_plot(plt, history_plot_filename)

    hl = hamming_loss(validation_y, y_pred)
    sa = accuracy_score(validation_y, y_pred)
    micro_f1 = f1_score(validation_y, y_pred, average='micro')
    macro_f1 = f1_score(validation_y, y_pred, average='macro')
    weighted_f1 = f1_score(validation_y, y_pred, average='weighted')
    auc_pr = average_precision_score(validation_y, y_pred, average='micro')

    eval_loss = eval_results[0]
    eval_accuracy = eval_results[1]

    print_and_log(logger, f"Hamming Loss: {hl:.4f}")
    print_and_log(logger, f"Subset Accuracy: {sa:.4f}")
    print_and_log(logger, f"Micro F1 Score: {micro_f1:.4f}")
    print_and_log(logger, f"Macro F1 Score: {macro_f1:.4f}")
    print_and_log(logger, f"Weighted F1 Score: {weighted_f1:.4f}")
    print_and_log(logger, f"AUC-PR: {auc_pr:.4f}")   
    print_and_log(logger, f"Evaluate results: {eval_results}")
    print_and_log(logger, f"Evaluated Validation Loss: {eval_loss:.4f}")
    print_and_log(logger, f"Evaluated Validation Accuracy: {eval_accuracy:.4f}")

    return hl, sa, micro_f1, macro_f1, weighted_f1, auc_pr, eval_loss, eval_accuracy, validation_y, y_pred

def train_and_evaluate_model(create_crtnet_method, samples, one_hot_encoding_labels, classes, is_multilabel, initial_learning_rate, number_of_leads, callbacks=None, epochs=10, batch_size=64, folds=None):

    method_name = create_crtnet_method.__name__
    logger = setup_logging(method_name)  # Initialize logging

    if classes is None:
        classes = ["Class " + str(i) for i in range(one_hot_encoding_labels.shape[1])]

    if folds:
        not_test_x, test_x, not_test_y, test_y = train_test_split(samples, one_hot_encoding_labels, test_size=0.1, random_state=42)
        kf = KFold(n_splits=folds, shuffle=True, random_state=42)

        metrics = {
            'hl': [], 'sa': [], 'micro_f1': [], 'macro_f1': [], 'weighted_f1': [],
            'auc_pr': [], 'eval_loss': [], 'eval_accuracy': []
        }

        for fold, (train_idx, val_idx) in enumerate(kf.split(not_test_x)):
            model = create_crtnet_method(number_of_leads, one_hot_encoding_labels.shape[1], is_multilabel, initial_learning_rate)
            history = train_model(logger, model, not_test_x[train_idx], not_test_y[train_idx], not_test_x[val_idx], not_test_y[val_idx], epochs, batch_size, callbacks)

            hl, sa, micro_f1, macro_f1, weighted_f1, auc_pr, eval_loss, eval_accuracy, validation_y, y_pred = evaluate_model(logger, model, not_test_x[val_idx], not_test_y[val_idx], is_multilabel, classes, method_name, history, epochs, fold)

            for metric, value in zip(metrics.keys(), [hl, sa, micro_f1, macro_f1, weighted_f1, auc_pr, eval_loss, eval_accuracy]):
                metrics[metric].append(value)

            model_filename = generate_filename('model', method_name, is_multilabel, 'h5', fold)
            #model.save(model_filename)
            #print_and_log(logger, f"Model saved to {model_filename}")

        metrics_mean = {metric: np.mean(values) for metric, values in metrics.items()}
        metrics_std = {metric: np.std(values) for metric, values in metrics.items()}
        
        print_and_log(logger, "Metrics (Mean ± Std Dev):")
        for metric in metrics_mean:
            print_and_log(logger, f"{metric}: {metrics_mean[metric]:.4f} ± {metrics_std[metric]:.4f}")

    else:
        train_x, val_x, train_y, val_y = train_test_split(samples, one_hot_encoding_labels, test_size=0.1, random_state=42)
        model = create_crtnet_method(number_of_leads, one_hot_encoding_labels.shape[1], is_multilabel, initial_learning_rate)
        history = train_model(logger, model, train_x, train_y, val_x, val_y, epochs, batch_size, callbacks)

        hl, sa, micro_f1, macro_f1, weighted_f1, auc_pr, eval_loss, eval_accuracy, validation_y, y_pred = evaluate_model(logger, model, val_x, val_y, is_multilabel, classes, method_name, history, epochs)

        model_filename = generate_filename('model', method_name, is_multilabel, 'h5')
        #model.save(model_filename)
        #print_and_log(logger, f"Model saved to {model_filename}")

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
