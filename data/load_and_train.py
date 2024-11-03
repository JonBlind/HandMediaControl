import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras import layers, models, callbacks, regularizers


def load_data_numpy(preprocessed_directory):
    '''
        <p>Loads the all the data found in the given directory into a set of numpy arrays.</p>
        
        <pre>load_data_numpy(preprocessed_directorys)</pre>
    
        <strong>Arguments:</strong>
        <ul>
            <li><code>preprocessed_directorys</code>, <code>PATH</code> of the directory relative to this file that contains .JSON files consisting of preprocessed gesture data.</li>
        </ul>

        <strong>Return:</strong>\n
            X : <code>load_data_numpy()[0]</code>: Data Array Containing [landmarks, handedness, wrist_displacement]. </p>
            Y : <code>load_data_numpy()[1]</code>: Label Array containing the corresponding results/labels for each data set in X.
        '''



    X, Y = [], []

    # For each file(.JSON) in the directory, read each file.
    # Add all the sequences in the file to an array
    # For each sequence, we extract the landmark info, handedness, and the wrist displacement.
    # Append each sequence's extracted info to X, and each gesture to Y.
    for filename in os.listdir(preprocessed_directory):
        if filename.endswith('.json'):
            with open(os.path.join(preprocessed_directory, filename), 'r') as file:
                data = json.load(file)

                for sequence in data:
                    sequence_data = []

                    for frame in sequence["sequence_data"]:

                        # Use 1 for "Right" and 0 for "Left" in handedness
                        handedness = [1] if frame["handedness"] == "Right" else [0]
                        wrist_displacement = frame["wrist_displacement"]
                        absolute_landmarks = frame["landmarks"]
                        relative_landmarks = frame["relative_landmarks"]

                        frame_features = absolute_landmarks + relative_landmarks + handedness + wrist_displacement
                        sequence_data.append(frame_features)

                    X.append(sequence_data)
                    Y.append(sequence["gesture"])
    return np.array(X), np.array(Y)

def prepare_labels(Y):
    '''
        <p>Manipulates a given Label Array, formatting it for the learning model.\n
        Each label in the vector will transform to a corresponding index, and the vector will turn to a one_hot matrix.</p>
        
        <pre>prepare_labels(Y : Numpy Label Array)</pre>
    
        <strong>Arguments:</strong>
        <ul>
            <li><code>Y</code>, <code>Numpy Array</code> consisting of labels that have been preprocessed and loaded through the proper method.</li>
        </ul>

        <strong>Return:</strong>\n
            Np.Array : <code>prepare_labels()[1]</code>: Numpy array of a one_hot matrix representing the inputted label vector.
        '''
    
    # Grab all different labels, and put them in a set
    label_set = sorted(set(Y))

    # For each label in the set, allocate them a number to represent them.
    label_map = {}
    for idx, label in enumerate(label_set):
        label_map[label] = idx

    # For each gesture in Y, make an array of corresponding gesture_labels in their numbered form.
    Y_numbered = []
    for gesture in Y:
        Y_numbered.append(label_map[gesture])

    # Convert the new label array to a numpy array.
    Y_numbered = np.array(Y_numbered)

    Y_one_hot = []
    num_classes = len(label_set)
    for number_label in Y_numbered:
        vector = [0] * num_classes
        vector[number_label] = 1
        Y_one_hot.append(vector)

    return np.array(Y_one_hot), label_map

def split_data(X, Y):
    '''
    Using Sklearn's train_test_split, split the given numpy data and label arrays by 70% training, 15% to validation, and 15% to testing.
    '''

    # 70% to training, 15% to validation, and 15% to testing.
    X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.3, random_state=42)
    X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

    return X_train, X_val, X_test, Y_train, Y_val, Y_test


def create_model(sequence_length, num_features, num_classes):
    '''
        <p> Method to build a tensorflow learning model given the length of a sequence, and number of elements in a frame and in the label_set.</p>
        
        <pre>create_model(sequence_length, num_features, num_classes)</pre>
    
        <strong>Arguments:</strong>
        <ul>
            <li><code>sequence_length</code>, <code>Int</code>Number of frames that each sequence has information for.</li>
            <li><code>num_features</code>, <code>Int</code>Number of features that each frame tracks. (63 landmark coords + 3 wrist wrist displacement coords + 1 handedness = 67)</li>
            <li><code>num_classes</code>, <code>Int</code>Number of possible gestures/labels. Length of label_set.</li>
        </ul>

        <strong>Return:</strong>\n
        Model : TensorFlow learning model that uses Long Short-Term Memory layers and Dense layers for output.\n
        The model calculates loss via cross-entropy and follows the "adam" optimizer. (Only one I'm familiar with tbh.)
        '''

    # Basically saw a large mix of videos and pages saying using a Neural network following a Long short-term Memory
    # model would be ideal for optical detection. Also heavily inspired from AI lab I did. 
    model = models.Sequential([
        layers.Bidirectional(layers.LSTM(128, return_sequences=True), input_shape=(sequence_length, num_features)),
        layers.Dropout(0.3),
        layers.LSTM(128),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu', kernel_regularizer= regularizers.l2(0.03)),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(X_train, X_val, Y_train, Y_val, sequence_length, num_features, num_classes):
    '''
        <p>Method to train the learning model. Grabs the output of the sklearn split, specified parameters to pass into create_model, \n
        and uses them to create a model, train it, and plot its accuracy and loss over epochs. (Default: 500 epochs, 100 batch_size).\n
        THIS WILL SAVE A COPY OF THE MODEL AND THE ACCURACY/LOSS GRAPH IN THE ./data/model/ DIRECTORY.</p>
        
        <pre>train_model(X_train, X_val, Y_train, Y_val, sequence_length, num_features, num_classes)</pre>
    
        <strong>Arguments:</strong>
        <ul>
            <li><code>X_train</code>, <code>NumPy Array</code>Split_Data[0] output, outputting the 70% of the data set to train.</li>
            <li><code>X_val</code>, <code>NumPy Array</code>Split_Data[1] output, outputting the 15% of the data set for validation.</li>
            <li><code>Y_train</code>, <code>NumPy Array</code>Split_Data[3] output, outputting the 70% of the label set to train.</li>
            <li><code>Y_val</code>, <code>NumPy Array</code>Split_Data[4] output, outputting the 15% of the label set to validate.</li>
            <li><code>sequence_length</code>, <code>Int</code>Number of frames that each sequence has information for.</li>
            <li><code>num_features</code>, <code>Int</code>Number of features that each frame tracks. (63 landmark coords + 3 wrist wrist displacement coords + 1 handedness = 67)</li>
            <li><code>num_classes</code>, <code>Int</code>Number of possible gestures/labels. Length of label_set.</li>
        </ul>

        <strong>Return:</strong>\n
        Model: Saves a copy of the trained model into the ./data/model directory. Also returns the model in the method.
        '''
    
    model = create_model(sequence_length, num_features, num_classes)

    # Train the model and capture the training history

    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=170, restore_best_weights=True)
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.05, patience=55, min_lr=1e-5)

    history = model.fit(
        X_train, Y_train, 
        validation_data=(X_val, Y_val), 
        epochs=750, 
        batch_size= 65,
        callbacks=[early_stopping, reduce_lr])
    model.save(os.path.join('model', "gesture_model.keras"))

    # Plot loss and accuracy over epochs
    plt.figure(figsize=(14, 8))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.savefig(os.path.join('model', "accuracy_plot.png"))

    plt.show()

    return model

def save_label_map(label_map, directory="model"):
    """
    Saves the label map to a JSON file in the specified directory.

    Args:
        label_map (dict): Dictionary mapping class labels to their respective integer encoding.
        directory (str): Directory to save the label map JSON file.
    """
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "label_map.json"), "w") as file:
        json.dump(label_map, file, indent=4)

    print(f"Label map saved to {os.path.join(directory, 'label_map.json')}")

if __name__ == "__main__":
    '''
    Main method of the load_and_train file.
    This essentially creates the model and the graph for its training data in the ./model directory.
    Will print ending test accuracy.
    '''
    preprocessed_dir = 'gesture_data_preprocessed' 
    sequence_length = 30 

    # Load, prepare, and split data
    X, Y = load_data_numpy(preprocessed_dir)
    Y, label_map = prepare_labels(Y)
    X_train, X_val, X_test, Y_train, Y_val, Y_test = split_data(X, Y)

    # Define input dimensions
    sequence_length = X_train.shape[1]
    num_features = X_train.shape[2]
    num_classes = Y_train.shape[1]

    # Train the model
    model = train_model(X_train, X_val, Y_train, Y_val, sequence_length, num_features, num_classes)

    # Save label map as JSON file
    save_label_map(label_map)

    # Evaluate on test set
    test_loss, test_acc = model.evaluate(X_test, Y_test)
    print(f"Test accuracy: {test_acc}")
        



    