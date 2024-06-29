import gradio as gr
import json
import prepro
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model



def showData(predictions):

    num_to_category = {0: 'Gerade', 1: 'Kinnhaken', 2: 'Kopfhaken'}

    predicted_classes = np.argmax(predictions, axis=1)

    class_counts = np.bincount(predicted_classes)
    gerade_count = class_counts[0]
    kinnhaken_count = class_counts[1]
    kopfhaken_count = class_counts[2]
    total = gerade_count + kinnhaken_count + kopfhaken_count

    return gerade_count, kinnhaken_count, kopfhaken_count, total


def neuronalnet(x_train, y_train):
    mymodel = load_model('cnn.keras')
    dings = mymodel.predict(x_train)
    print(dings[1])
    return showData(dings)


def open_json(json_file):
    periodLengthMS = 1000
    sampleRateUS = 10000

    with open(json_file, 'r') as f:
        dsds = json.load(f)

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

    return neuronalnet(_x_train, _y_train)


with gr.Blocks() as demo:
    gr.Markdown("### Werte Dein Boxtraining aus!")
    json_input = gr.File(label="JSON Datei hochladen")

    with gr.Row():
        value1_output = gr.Textbox(label="Geraden", elem_id="Geraden")
        value2_output = gr.Textbox(label="Kopfhaken", elem_id="Kopfhaken")
        value3_output = gr.Textbox(label="Kinnhaken", elem_id="Kinnhaken")

    gr.Markdown("Tolles Training!")
    text = gr.Textbox(label="Total", elem_id="Total")

    json_input.change(open_json, inputs=json_input, outputs=[value1_output, value2_output, value3_output, text])

if __name__ == "__main__":
    demo.launch()
