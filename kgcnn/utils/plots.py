import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import os


def plot_train_test_loss(histories: list, loss_name: str = None,
                         val_loss_name: str = None, data_unit: str = "", model_name: str = "",
                         filepath: str = None, file_name: str = "", dataset_name: str = "",
                         figsize: list = None, dpi: float = None, show_fig: bool = True
                         ):
    r"""Plot training curves for a list of fit results in form of keras history objects. This means, training-
    and test-loss is plotted vs. epochs for all splits.

    Args:
        histories (list): List of :obj:`tf.keras.callbacks.History()` objects.
        loss_name (str): Which loss or metric to pick from history for plotting. Default is "loss".
        val_loss_name (str): Which validation loss or metric to pick from history for plotting. Default is "val_loss".
        data_unit (str): Unit of the loss. Default is "".
        model_name (str): Name of the model. Default is "".
        filepath (str): Full path where to save plot to, without the name of the file. Default is "".
        file_name (str): File name base. Model name and dataset will be added to the name. Default is "".
        dataset_name (str): Name of the dataset which was fitted to. Default is "".
        figsize (list): Size of the figure. Default is None.
        dpi (float): The resolution of the figure in dots-per-inch. Default is None.
        show_fig (bool): Whether to show figure. Default is True.

    Returns:
        matplotlib.pyplot.figure: Figure of the training curves.
    """
    X_MIN = 5
    histories = [hist.history if isinstance(hist, tf.keras.callbacks.History) else hist for hist in histories]
    # We assume multiple fits as in KFold.
    if data_unit is None:
        data_unit = ""
    if loss_name is None:
        loss_name = [x for x in list(histories[0].keys()) if "val_" not in x]
    if val_loss_name is None:
        val_loss_name = [x for x in list(histories[0].keys()) if "val_" in x]

    if not isinstance(loss_name, list):
        loss_name = [loss_name]
    if not isinstance(val_loss_name, list):
        val_loss_name = [val_loss_name]

    train_loss = []
    for x in loss_name:
        loss = np.array([np.array(hist[x]) for hist in histories])
        train_loss.append(loss)
    val_loss = []
    for x in val_loss_name:
        loss = np.array([hist[x] for hist in histories])
        val_loss.append(loss)

    if figsize is None:
        figsize = [6.4, 4.8]
    if dpi is None:
        dpi = 100.0
    fig = plt.figure(figsize=figsize, dpi=dpi)
    for i, x in enumerate(train_loss):
        vp = plt.plot(np.arange(len(np.mean(x, axis=0))), np.mean(x, axis=0), alpha=0.85, label=loss_name[i])
        plt.fill_between(np.arange(len(np.mean(x, axis=0))),
                         np.mean(x, axis=0) - np.std(x, axis=0),
                         np.mean(x, axis=0) + np.std(x, axis=0), color=vp[0].get_color(), alpha=0.2
                         )
    for i, y in enumerate(val_loss):
        val_step = len(train_loss[i][0]) / len(val_loss[i][0])
        vp = plt.plot(np.arange(len(np.mean(y, axis=0))) * val_step + val_step, np.mean(y, axis=0), alpha=0.85,
                      label=val_loss_name[i])
        plt.fill_between(np.arange(len(np.mean(y, axis=0))) * val_step + val_step,
                         np.mean(y, axis=0) - np.std(y, axis=0),
                         np.mean(y, axis=0) + np.std(y, axis=0), color=vp[0].get_color(), alpha=0.2
                         )
        plt.scatter([len(train_loss[i][0])], [np.mean(y, axis=0)[-1]],
                    label=r"{0}: {1:0.4f} $\pm$ {2:0.4f} ".format(
                        val_loss_name[i], np.mean(y, axis=0)[-1],
                        np.std(y, axis=0)[-1]) + data_unit, color=vp[0].get_color()
                    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    val_loss_array = np.stack(val_loss, axis=0)
    if val_loss_array.shape[2] <= X_MIN:
        X_MIN = 0
    plt.xlim(left=X_MIN)
    plt.ylim(top=np.max(np.percentile(val_loss_array[:, :, X_MIN:], 75, axis=0)), bottom=0)
    plt.title(dataset_name + " training curve for " + model_name)
    plt.legend(loc='upper right', fontsize='small')
    if filepath is not None:
        plt.savefig(os.path.join(filepath, model_name + "_" + dataset_name + "_" + file_name))
    if show_fig:
        plt.show()
    return fig


def plot_predict_true(y_predict, y_true, data_unit: list = None, model_name: str = "",
                      filepath: str = None, file_name: str = "", dataset_name: str = "", target_names: list = None,
                      error: str = "MAE", figsize: list = None, dpi: float = None, show_fig: bool = False):
    r"""Make a scatter plot of predicted versus actual targets. Not for k-splits.

    Args:
        y_predict (np.ndarray): Numpy array of shape `(N_samples, n_targets)` or `(N_samples, )`.
        y_true (np.ndarray): Numpy array of shape `(N_samples, n_targets)` or `(N_samples, )`.
        data_unit (list): String or list of string that matches `n_targets`. Name of the data's unit.
        model_name (str): Name of the model. Default is "".
        filepath (str): Full path where to save plot to, without the name of the file. Default is "".
        file_name (str): File name base. Model name and dataset will be added to the name. Default is "".
        dataset_name (str): Name of the dataset which was fitted to. Default is "".
        target_names (list): String or list of string that matches `n_targets`. Name of the targets.
        error (str): String, "MAE" or "RMSE". Error plotted on figure. Default is "MAE"
        figsize (list): Size of the figure. Default is None.
        dpi (float): The resolution of the figure in dots-per-inch. Default is None.
        show_fig (bool): Whether to show figure. Default is True.

    Returns:
        matplotlib.pyplot.figure: Figure of the scatter plot.
    """
    if len(y_predict.shape) == 1:
        y_predict = np.expand_dims(y_predict, axis=-1)
    if len(y_true.shape) == 1:
        y_true = np.expand_dims(y_true, axis=-1)
    num_targets = y_true.shape[1]

    if data_unit is None:
        data_unit = ""
    if isinstance(data_unit, str):
        data_unit = [data_unit]*num_targets
    if len(data_unit) != num_targets:
        print("WARNING:kgcnn: Targets do not match units for plot.")
    if target_names is None:
        target_names = ""
    if isinstance(target_names, str):
        target_names = [target_names]*num_targets
    if len(target_names) != num_targets:
        print("WARNING:kgcnn: Targets do not match names for plot.")
    if error is None:
        error = "MAE"
    elif error not in ("MAE", "RMSE"):
        print("WARNING:kgcnn: Unrecognized error type. Using MAE instead")
        error = "MAE"

    if figsize is None:
        figsize = [6.4, 4.8]
    if dpi is None:
        dpi = 100.0
    fig = plt.figure(figsize=figsize, dpi=dpi)
    for i in range(num_targets):
        delta_valid = y_true[:, i] - y_predict[:, i]
        mae_valid = np.mean(np.abs(delta_valid[~np.isnan(delta_valid)]))
        rmse_valid = np.sqrt(np.mean((delta_valid[~np.isnan(delta_valid)])**2))
        if error == "MAE":
            plt.scatter(y_predict[:, i], y_true[:, i], alpha=0.3,
                        label=target_names[i] + " MAE: {0:0.4f} ".format(mae_valid) + "[" + data_unit[i] + "]")
        elif error == "RMSE":
             plt.scatter(y_predict[:, i], y_true[:, i], alpha=0.3,
                        label=target_names[i] + " RMSE: {0:0.4f} ".format(rmse_valid) + "[" + data_unit[i] + "]")       
    min_max = np.amin(y_true[~np.isnan(y_true)]), np.amax(y_true[~np.isnan(y_true)])
    plt.plot(np.arange(*min_max, 0.05), np.arange(*min_max, 0.05), color='red')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title("Prediction of " + model_name + " for " + dataset_name)
    plt.legend(loc='upper left', fontsize='x-large')
    if filepath is not None:
        plt.savefig(os.path.join(filepath, model_name + "_" + dataset_name + "_" + file_name))
    if show_fig:
        plt.show()
    return fig

def plot_test_set_prediction(data: pd.DataFrame, observation: str, prediction: str, title: str = "", unit: str = "",
                             rmse: float = None, r2: float = None, filepath: str = None, show_fig: bool = False):
    plt.figure(figsize=(7,5))
    FONTSIZE  = 18
    LABELSIZE = 15
    if "at_types" in data.columns:
        plot = sns.lmplot(y=observation, x=prediction, data=data, hue="at_types",
            scatter_kws={"marker": ".", "alpha": 0.1},
            line_kws={"linewidth": 3},
            height=6,
            legend=False)
        legend = plt.legend(title='Atom Types', loc='upper left', fontsize=FONTSIZE, title_fontsize=FONTSIZE)

        for legend_handle in legend.legendHandles: 
            legend_handle.set_alpha(1)
    else:
        plot = sns.lmplot(y=observation, x=prediction, data=data,
            scatter_kws={"marker": ".", "alpha": 0.1},
            line_kws={"linewidth": 1},
            height=6)
    plt.xlabel(f"Prediction {title} [{unit}]", fontsize=FONTSIZE)
    plt.ylabel(f"Reference {title} [{unit}]",fontsize=FONTSIZE)
    value_min = data[[observation,prediction]].min().min()
    value_max = data[[observation,prediction]].max().max()
    plt.plot([value_min-1, value_max+1], [value_min-1, value_max+1], "k")
    plot.set(xlim=(value_min, value_max))
    plot.set(ylim=(value_min, value_max))
    plt.grid(which="major", linestyle="-", linewidth="0.5", color="purple")
    plt.grid(which="minor", linestyle="-", linewidth="0.5", color="blue", alpha=0.25)
    #plt.minorticks_on()
    plt.tick_params(axis='both', which="major", labelsize=LABELSIZE)
    #plt.title(f"Prediction {title}", fontsize=FONTSIZE)
    text_x = 0.75*(value_max-value_min) + value_min
    text_y = 0.1*(value_max-value_min) + value_min
    if rmse is not None and r2 is not None:
        plot.ax.text(text_x, text_y, f"RMSE: {rmse:.3f}\nR2: {r2:.2f}", bbox={
        "facecolor": "grey", "alpha": 0.5, "pad": 10})
    elif rmse is not None:
        plot.ax.text(text_x, text_y, f"RMSE: {rmse:.3f}", bbox={
        "facecolor": "grey", "alpha": 0.5, "pad": 10})
    elif r2 is not None:
        plot.ax.text(text_x, text_y, f"R2: {r2:.2f}", bbox={
        "facecolor": "grey", "alpha": 0.5, "pad": 10})
    if filepath is not None:
        plt.savefig(os.path.join(filepath, f"{'_'.join(title.split())}_lmplot"), dpi=300, bbox_inches = "tight")
    if show_fig is True:
        plt.show()
    plt.close()