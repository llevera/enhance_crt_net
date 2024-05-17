from tensorflow.keras import layers
import keras_nlp as nlp
import tensorflow.keras as keras
from sklearn.metrics import classification_report, multilabel_confusion_matrix, accuracy_score, f1_score, confusion_matrix
from sklearn.metrics import hamming_loss, average_precision_score
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
import io
import contextlib

# Setting seeds for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

def setup_logging(method_name, is_multilabel, folds=None):
    log_filename = generate_filename("log", method_name, is_multilabel, "log", folds)
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

def print_and_log_model(logger, model):
    summary_buffer = io.StringIO()
    with contextlib.redirect_stdout(summary_buffer):
        model.summary()
    model_summary_str = summary_buffer.getvalue()
    print_and_log(logger, model_summary_str)

def train_model(logger, model, train_x, train_y, validation_x, validation_y, epochs, batch_size, callbacks):
    history = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, validation_data=(validation_x, validation_y), callbacks=callbacks, verbose=1)
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

    report = classification_report(validation_y, y_pred, target_names=classes, zero_division=0)
    print_and_log(logger, f"Model Method: {method_name}\n{report}")

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
        plt.title(f'Confusion Matrix')
        plot_filename = generate_filename('confusion_matrix', method_name, is_multilabel, 'png', fold)
        save_and_show_plot(plt, plot_filename)
    
    accuracy_logs = "\n".join([f"Class '{class_name}' Accuracy: {accuracy:.4f}" for class_name, accuracy in per_class_accuracies_dict.items()])
    print_and_log(logger, accuracy_logs)

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

    if validation_y.ndim == 2 and y_pred.ndim == 2:
        auc_pr = average_precision_score(validation_y, y_pred, average='micro')
    else:
        auc_pr = -1
        print_and_log(logger, "Skipping AUC-PR calculation because validation_y or y_pred is not 2D")

    eval_loss = eval_results[0]
    eval_accuracy = eval_results[1]

    eval_metrics = {
        'hamming_loss': hl,
        'subset_accuracy': sa,
        'micro_f1': micro_f1,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'auc_pr': auc_pr,
        'eval_loss': eval_loss,
        'eval_accuracy': eval_accuracy
    }

    for metric, value in eval_metrics.items():
        print_and_log(logger, f"{metric.replace('_', ' ').title()}: {value:.4f}")

    return eval_metrics, validation_y, y_pred, history.history

def train_and_evaluate_model(create_crtnet_method, samples, one_hot_encoding_labels, classes, is_multilabel, initial_learning_rate, number_of_leads, callbacks=None, epochs=10, batch_size=64, folds=None):
    method_name = create_crtnet_method.__name__
    logger = setup_logging(method_name, is_multilabel, folds)

    if classes is None:
        classes = ["Class " + str(i) for i in range(one_hot_encoding_labels.shape[1])]

    all_folds_data = []

    if folds:
        not_test_x, test_x, not_test_y, test_y = train_test_split(samples, one_hot_encoding_labels, test_size=0.1, random_state=42)
        kf = KFold(n_splits=folds, shuffle=True, random_state=42)

        for fold, (train_idx, val_idx) in enumerate(kf.split(not_test_x)):
            model = create_crtnet_method(number_of_leads, one_hot_encoding_labels.shape[1], is_multilabel, initial_learning_rate)
            print_and_log_model(logger, model)

            history = train_model(logger, model, not_test_x[train_idx], not_test_y[train_idx], not_test_x[val_idx], not_test_y[val_idx], epochs, batch_size, callbacks)

            eval_metrics, val_y, pred_y, history_data = evaluate_model(logger, model, not_test_x[val_idx], not_test_y[val_idx], is_multilabel, classes, method_name, history, epochs, fold)

            fold_data = {
                'fold': fold,
                'evaluation_metrics': eval_metrics,
                'validation_labels': val_y.tolist(),
                'predicted_labels': pred_y.tolist(),
                'training_history': history_data
            }
            all_folds_data.append(fold_data)

        metrics_data = {metric: [fold_data['evaluation_metrics'][metric] for fold_data in all_folds_data] for metric in all_folds_data[0]['evaluation_metrics'].keys()}

        metrics_mean = {metric: np.mean(values) for metric, values in metrics_data.items()}
        metrics_std = {metric: np.std(values) for metric, values in metrics_data.items()}

        training_run_summary = {
            'mean_metrics': metrics_mean,
            'std_metrics': metrics_std
        }
        print_and_log(logger, "Metrics (Mean ± Std Dev):")
        for metric in metrics_mean:
            print_and_log(logger, f"{metric}: {metrics_mean[metric]:.4f} ± {metrics_std[metric]:.4f}")

        # Box plot for each metric
        metrics_df = pd.DataFrame(metrics_data)
        metrics_df = metrics_df.melt(var_name='Metric', value_name='Score')
        plt.figure(figsize=(12, 8))
        sns.boxplot(x='Metric', y='Score', data=metrics_df)
        plt.title('Evaluation Metrics Across Folds')
        plt.xticks(rotation=45)
        plt.tight_layout()
        boxplot_filename = generate_filename('boxplot_metrics', method_name, is_multilabel, 'png')
        save_and_show_plot(plt, boxplot_filename)

    else:
        train_x, val_x, train_y, val_y = train_test_split(samples, one_hot_encoding_labels, test_size=0.1, random_state=42)
        model = create_crtnet_method(number_of_leads, one_hot_encoding_labels.shape[1], is_multilabel, initial_learning_rate)
        print_and_log_model(logger, model)

        history = train_model(logger, model, train_x, train_y, val_x, val_y, epochs, batch_size, callbacks)

        eval_metrics, val_y, pred_y, history_data = evaluate_model(logger, model, val_x, val_y, is_multilabel, classes, method_name, history, epochs)

        fold_data = {
            'fold': 0,
            'evaluation_metrics': eval_metrics,
            'validation_labels': val_y.tolist(),
            'predicted_labels': pred_y.tolist(),
            'training_history': history_data
        }
        all_folds_data.append(fold_data)

        metrics_mean = {metric: eval_metrics[metric] for metric in eval_metrics.keys()}
        metrics_std = {metric: 0 for metric in eval_metrics.keys()}  # Standard deviation is zero since there's only one fold

        training_run_summary = {
            'mean_metrics': metrics_mean,
            'std_metrics': metrics_std
        }
        print_and_log(logger, "Metrics (Mean ± Std Dev):")
        for metric in metrics_mean:
            print_and_log(logger, f"{metric}: {metrics_mean[metric]:.4f} ± {metrics_std[metric]:.4f}")

    return {
        'folds_data': all_folds_data,
        'training_run_summary': training_run_summary
    }

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
