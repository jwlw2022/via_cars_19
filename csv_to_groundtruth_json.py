import json
import pandas as pd

# change path
csv_path = "test_groundtruth.csv"
df = pd.read_csv(csv_path)

# change according to number of rows in csv 
df = pd.DataFrame(df, index=[x for x in range(1142)])

gt_dict = {}

for image in df['image_path']:
    image = image.replace('datasets/test/', '')
    image = image.replace('.jpg', '')
    gt_dict.update({image: []})

counter = 0
for key in gt_dict:
    gt_dict[key].append(int(df.loc[ counter , : ]['xmin']))
    gt_dict[key].append(int(df.loc[ counter , : ]['ymin']))
    gt_dict[key].append(int(df.loc[ counter , : ]['xmax']))
    gt_dict[key].append(int(df.loc[ counter , : ]['ymax']))
    counter += 1

print(gt_dict)

    
# write to file - change path name
with open('groundtruth.json', 'w') as json_file:
    json.dump(gt_dict, json_file)