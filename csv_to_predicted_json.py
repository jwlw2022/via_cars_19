import json
import pandas as pd

# change path
csv_path = "test_ssd_resnet50_2.csv"
df = pd.read_csv(csv_path)

# change according to number of rows in csv 
df = pd.DataFrame(df, index=[x for x in range(1141)])

predicted_dict = {}

for image in df['image_path']:
    predicted_dict.update({image: {}})

counter = 0
for key in predicted_dict:
    predicted_dict[key].update({"boxes": []})
    predicted_dict[key].update({"scores": []})
    predicted_dict[key]['boxes'].append([])
    predicted_dict[key]['boxes'][0].append(int(df.loc[ counter , : ]['xmin']))
    predicted_dict[key]['boxes'][0].append(int(df.loc[ counter , : ]['ymin']))
    predicted_dict[key]['boxes'][0].append(int(df.loc[ counter , : ]['xmax']))
    predicted_dict[key]['boxes'][0].append(int(df.loc[ counter , : ]['ymax']))
    predicted_dict[key]['scores'].append(df.loc[ counter , : ]['Confidence'])
    counter += 1

print(predicted_dict)

    
# write to file - change path name
with open('predicted.json', 'w') as json_file:
    json.dump(predicted_dict, json_file)