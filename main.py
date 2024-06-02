from data_preprocessing import extract_accelerometer_data, csv_to_dataset_list, csv_to_json
import pandas as pd

data = csv_to_json(r'C:\Users\Raoul\Documents\GitHub\Boxschlag-Klassifikation\data\Max_Gerade.csv')
ds = extract_accelerometer_data(r'C:\Users\Raoul\Documents\GitHub\Boxschlag-Klassifikation\data\Max_Gerade.csv')

#print(ds)
print(data)

