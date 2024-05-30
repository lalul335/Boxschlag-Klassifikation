from data_preprocessing import extract_accelerometer_data, csv_to_dataset_list
import pandas as pd

data = extract_accelerometer_data('/Users/raouldoublan/Documents/GitHub/Boxschlag-Klassifikation/data/Max_Gerade.csv')

print(data)

# Assuming 'data' is your DataFrame
raw_data = data.to_dict('records')

# Now, 'raw_data' is a list of dictionaries where each dictionary represents a row in 'data'
# You can put this list under the key 'raw' in another dictionary

final_dict = {'raw': raw_data}

#print(final_dict)

