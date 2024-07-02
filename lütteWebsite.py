import gradio as gr
import json
import prepro
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import prepro


def plotdata(_x_train, predictions):
    hoechster_wert_0 = max(predictions, key=lambda x: x[0])[0]
    hoechster_wert_1 = max(predictions, key=lambda x: x[1])[1]
    hoechster_wert_2 = max(predictions, key=lambda x: x[2])[2]
    index_hoechster_wert_0 = np.argmax(predictions[:, 0])
    index_hoechster_wert_1 = np.argmax(predictions[:, 1])
    index_hoechster_wert_2 = np.argmax(predictions[:, 2])

    def plot_instance(ax, instance, title):
        FEATURE_NAMES = ['x', 'y', 'z']

        # Plot the feature groups (all axis of one measurement) in the different subplots and add labels
        for subplot in [0, 1, 2]:
            ax[subplot].plot(instance[:, subplot])
            ax[subplot].legend([FEATURE_NAMES[subplot]])

        ax[0].set_title(title)

    fig, axs = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=False)
    fig.suptitle('Visualization of instances', fontsize=16)
    fig.set_size_inches(16, 9)

    # Plot _x_train[9]
    plot_instance(axs[:, 0], _x_train[index_hoechster_wert_0], 'Gerade')

    # Plot _x_train[8]
    plot_instance(axs[:, 1], _x_train[index_hoechster_wert_1], 'Kopfhaken')

    # Plot _x_train[7]
    plot_instance(axs[:, 2], _x_train[index_hoechster_wert_2], "Kinnhaken")

    plt.savefig('plot.png')
    return 'plot.png'


def showData(x_train, predictions):
    num_to_category = {0: 'Gerade', 1: 'Kopfhaken', 2: 'Kinnhaken'}

    predicted_classes = np.argmax(predictions, axis=1)

    class_counts = np.bincount(predicted_classes)
    gerade_count = class_counts[0]

    kopfhaken_count = class_counts[1]
    kinnhaken_count = class_counts[2]
    total = gerade_count + kinnhaken_count + kopfhaken_count
    plot = plotdata(x_train, predictions)
    return gerade_count, kopfhaken_count, kinnhaken_count, total, plot


def neuronalnet(x_train, y_train):
    mymodel = load_model('cnn.keras')
    dings = mymodel.predict(x_train)
    print(dings[1])
    return showData(x_train, dings)


def open_json(dsds):
    periodLengthMS = 1000
    sampleRateUS = 10000

    ds = prepro.jsonData_to_dataset_in_timedifference_us(dsds)

    _df_new = prepro.normate_dataset_period(periodLengthMS, sampleRateUS, ds)

    _df_list = pd.DataFrame({'idx': range(len(_df_new)), 'dfs': _df_new})['dfs'].values.tolist()

    _x_train, _y_train = [], []

    for df in _df_list:
        _y_train.append(df["label"].iloc[0])
        _x_train.append(df.drop(columns=["timestamp", "label"]).values)

    _x_train = np.array(_x_train)
    _y_train = np.array(_y_train)

    _label = ['Gerade', 'Kopfhaken', 'Kinnhaken']
    category_to_num = {element: num for num, element in enumerate(_label)}
    numerical_data = np.vectorize(category_to_num.get)(_y_train)

    _y_train = np.array(numerical_data)

    return neuronalnet(_x_train, _y_train)


def labeling(data, schwelle, grenze):

    return open_json(prepro.auto_labeling(data, schwelle, grenze, "unknown"))


def csv_convert(csvfile, Schwelle, Grenze):

    return labeling((prepro.extract_accelerometer_data(csvfile)), Schwelle, Grenze)


def showData_beta(x_train, predictions):
    num_to_category = {0: 'Gerade', 1: 'Kopfhaken', 2: 'Kinnhaken'}

    predicted_classes = np.argmax(predictions, axis=1)

    class_counts = np.bincount(predicted_classes)
    gerade_count = class_counts[0]
    kopfhaken_count = class_counts[1]
    kinnhaken_count = class_counts[2]

    # Finden Sie den größten Wert jeder Zeile und den zugehörigen Index
    max_values = np.max(predictions, axis=1)
    max_indices = np.argmax(predictions, axis=1)

    # Erstellen Sie einen neuen Datensatz aus den größten Werten und den zugehörigen Indizes
    new_data = list(zip(max_values, max_indices))

    # Erstellen Sie eine neue Liste für Werte unter 0.6

    values_below_06 = [(index) for value, index in new_data if value < 0.6]

    def count_classes(values):
        # Initialisieren Sie ein Wörterbuch mit den Klassen 0, 1 und 2, die alle auf 0 gesetzt sind
        class_counts = {0: 0, 1: 0, 2: 0}

        # Zählen Sie die Häufigkeit jeder Klasse in der Liste
        for value in values:
            if value in class_counts:
                class_counts[value] += 1

        return class_counts

    #0 = gerade, 1= kinn, 2 = kopf
    counter_per_schlag = count_classes(values_below_06)

    return gerade_count, kopfhaken_count, kinnhaken_count, len(values_below_06), counter_per_schlag[0], \
        counter_per_schlag[1], counter_per_schlag[2]


def neuronalnet_beta(x_train, y_train):
    mymodel = load_model('cnn.keras')
    dings = mymodel.predict(x_train)
    print(dings[1])
    return showData_beta(x_train, dings)


def open_json_beta(dsds):
    periodLengthMS = 1000
    sampleRateUS = 10000

    ds = prepro.jsonData_to_dataset_in_timedifference_us(dsds)

    _df_new = prepro.normate_dataset_period(periodLengthMS, sampleRateUS, ds)

    _df_list = pd.DataFrame({'idx': range(len(_df_new)), 'dfs': _df_new})['dfs'].values.tolist()

    _x_train, _y_train = [], []

    for df in _df_list:
        _y_train.append(df["label"].iloc[0])
        _x_train.append(df.drop(columns=["timestamp", "label"]).values)

    _x_train = np.array(_x_train)
    _y_train = np.array(_y_train)

    _label = ['Gerade', 'Kinnhaken', 'Kopfhaken']
    category_to_num = {element: num for num, element in enumerate(_label)}
    numerical_data = np.vectorize(category_to_num.get)(_y_train)

    _y_train = np.array(numerical_data)

    return neuronalnet_beta(_x_train, _y_train)


def labeling_beta(data, schwelle, grenze):
    return open_json_beta(prepro.auto_labeling(data, schwelle, grenze, "unknown"))



def csv_convert_beta(csvfile, schwelle, grenze):

    return labeling_beta((prepro.extract_accelerometer_data(csvfile)), schwelle, grenze)



with gr.Blocks() as demo:
    with gr.Tab("Deine Boxanalyse"):
        gr.Markdown("### Werte Dein Boxtraining aus!")
        json_input = gr.File(label="JSON Datei hochladen")
        Schwelle = gr.Slider(0.5, 1.5, value=1, label="Schwelle", step=0.1, info="Grenze für Peaks", interactive=True)    #TODO
        Grenze = gr.Slider(80, 120, value=90, label="Grenzen", step=1, info="Startwert/Endwert für Peaks", interactive=True)
        gr.Markdown("Tolles Training!")

        with gr.Row():
            value1_output = gr.Textbox(label="Geraden", elem_id="Geraden")
            value2_output = gr.Textbox(label="Kopfhaken", elem_id="Kopfhaken")
            value3_output = gr.Textbox(label="Kinnhaken", elem_id="Kinnhaken")
        text = gr.Textbox(label="Total", elem_id="Total")
        plot_output = gr.Image(label="plotting data...")

        json_input.change(fn=csv_convert, inputs=[json_input, Schwelle, Grenze],
                          outputs=[value1_output, value2_output, value3_output, text, plot_output])

    with gr.Tab("Dein Box-Coach(Beta)"):

        gr.Markdown("Hey! Ich bin dein Box-Coach. Lade ein paar Schläge hoch und ich werte sie für dich aus!")
        json_input = gr.File(label="JSON Datei hochladen")                      # TODO
        Schwelle = gr.Slider(0.5, 1.5, value=1, label="Schwelle", info="Grenze für Peaks", interactive=True)
        Grenze = gr.Slider(80, 120, value=90, label="Grenzen", step=1, info="Startwert/Endwert für Peaks", interactive=True)
        with gr.Row():
            value1_output = gr.Textbox(label="Geraden", elem_id="Geraden")
            value2_output = gr.Textbox(label="Kopfhaken", elem_id="Kopfhaken")
            value3_output = gr.Textbox(label="Kinnhaken", elem_id="Kinnhaken")
        gr.Markdown("Das war schon nicht schlecht! So viele Schläge waren aber nicht so sauber:")
        text = gr.Textbox(label="unsaubere Schläge", elem_id="unsaubere Schläge")
        gr.Markdown("Diese Klassen habe ich dabei vermutet:")
        with gr.Row():
            text1 = gr.Textbox(label="unsaubere Geraden", elem_id="unsaubere Geraden")
            text2 = gr.Textbox(label="unsaubere Kopfhaken", elem_id="unsaubere Kopfhaken")
            text3 = gr.Textbox(label="unsaubere Kinnhaken", elem_id="unsaubere Kinnhaken")

        json_input.change(fn=csv_convert_beta, inputs=[json_input, Schwelle, Grenze],
                          outputs=[value1_output, value2_output, value3_output, text, plot_output])

if __name__ == "__main__":
    demo.launch()




