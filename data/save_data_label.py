import json

# This file is because I forgot to incorporate the actual lines to save the data label dictionary for the model.
# But I saved the output so I'm manually making the .JSON file here.


# Define your label map manually
label_map = {
    "open_palm": 0,
    "point_down": 1,
    "point_up": 2,
    "swipe_left": 3,
    "swipe_right": 4
}

# Save label map to a JSON file
with open('model/label_map.json', 'w') as file:
    json.dump(label_map, file, indent=4)

print("Label map saved to model/label_map.json")