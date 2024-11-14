import os
import json



class PreprocessGestureData:
    '''

    Class Definition: PreprocessGestureData

    This is a class designed for preprocessing data with far more ease when it comes to the gesture data.\n

    This is made in case there is a need to work with new data/information in future cases.   

    Arguments:
        source (PATH): file path of the JSON file to read information from.
        destination (PATH): file path of the JSON file (new or not) to deposit all the information into.
        sequence_length (INT): Number of frames that each sequence should take up.

    If the sequence_length of the frames is smaller than from the data set, we will choose the center frames and cut out the extreme ends.\n
    If the sequence_length of the frames is larger than from the data set, we will duplicate the extreme ends to fill up both ends equally until the target is reached.

    '''
    def __init__(self, source, destination, sequence_length):
        self.source = source
        self.destination = destination
        self.sequence_length = sequence_length

    
    def load_data(self):
        '''Method to grab the JSON file from source and extract all the data.'''
        with open(self.source, 'r') as file:
            data = json.load(file)
        return data
    
    
    def calculate_wrist_displacement(self, current_wrist, prior_wrist):
        '''
        Given a set of coordinates for the current wrists, and the prior wrists,\n
        will calcualte the displacement by subtracting the prior from the current position.
        
        Arguments:
            current_wrist (Int[3]): array of 3 integers represnting the x,y,z coordinates of the current wrist
            prior_wrist (Int[3]) : array of 3 integers represnting the x,y,z coordinates of the prior wrist
    

        Return:
            Array of 3 int, representing the difference in current by prior.

        '''
        if prior_wrist is None:
            return [0, 0, 0]
        else:
            return [
                current_wrist[0] - prior_wrist[0],
                current_wrist[1] - prior_wrist[1],
                current_wrist[2] - prior_wrist[2]
            ]
        
    def calculate_wrist_relative_landmarks(self, landmarks):
        '''
        Method to calculate the wrist-relative landmarks given a frame's landmarks.
        Essentially, will subtract every coordinate with the wrist's coordinate, almost centering around the wrist.
        '''
        wrist_x, wrist_y, wrist_z = landmarks[:3]
        relative_landmarks = []

        for i in range(0, len(landmarks), 3):
            x, y, z = landmarks[i : i+3]
            relative_landmarks.extend([
                x - wrist_x,
                y - wrist_y,
                z - wrist_z
            ])

        return relative_landmarks
    
    def pad_sequence(self, sequence):
        '''
        Given a sequence from the data_gathered JSON file that has less frames than desired, pad the sequence.\n
        Identifies the amount of frames needed, and then splits the amount between the front and end of the sequence,\n
        where the front and end frames are duplicated for their respective amount.
    
        Arguments:
            Sequence: List of frames containing information of the hand being tracked.

        Return:
            A padded sequence that has either end duplicated to retain natural flow and achieving the desired frame count.
        
        '''
    
        num_frames_needed = self.sequence_length - len(sequence)

        start_count = num_frames_needed // 2
        end_count = num_frames_needed - start_count

        padded_sequence = [sequence[0]] * start_count + sequence + [sequence[-1]] * end_count
        
        return self.reindex_sequence(padded_sequence)
    
    def trim_sequence(self, sequence):
        '''
        Given a sequence from the data_gathered JSON file that has less frames than desired, trim the sequence.\n
        Identifies the amount of frames needed to remove, and then splits the amount between the front and end of the sequence,\n
        where the front and end frames are removed for their respective amount.
    
        Arguments:
            Sequence (list of frames): List of frames containing information of the hand being tracked.

        Return:
            A trimmed sequence that has either end removed to retain natural flow and achieving the desired frame count.
        
        '''
        excess_frame_count = len(sequence) - self.sequence_length

        start_count = excess_frame_count // 2
        end_count = excess_frame_count - start_count

        trimmed_sequence = sequence[start_count:len(sequence) - end_count]
        
        return self.reindex_sequence(trimmed_sequence)
    
    def reindex_sequence(self, sequence):
        """
        Helper to reindex the frame indices in a sequence from 0 to sequence_length - 1.
        
        Arguments:
            sequence (list): List of frames in the sequence, each with a "frame_index".
            
        Returns:
            list: The sequence with updated frame indices.
        """

        for i, frame in enumerate(sequence):
            frame["frame_index"] = i
        return sequence

    

    def process_sequence(self, sequence_data):
        '''
        Given a sequence from the data_gathered JSON file that has less frames than desired, process the sequence.\n
        Will calculate the previous wrist position based on the prior frame.\n
        If needed, the sequence will be either trimmed or padded to maintain constant frame count across the data.\n
    
        Arguments:
            Sequence: List of frames containing information of the hand being tracked.

        Return:
        A sequence processed to have landmarks normalized about the wrist and the proper frame count.
        
        '''

        processed_sequence = []
        previous_wrist_pos = None

        # For each frame in the sequence, normalize/translate the landmarks
        # Then identify the wrist displacement.
        # Add the new/updated info to the processed array.
        for frame in sequence_data:
            wrist_x, wrist_y, wrist_z = frame["landmarks"][:3]
            wrist_displacement = self.calculate_wrist_displacement((wrist_x, wrist_y, wrist_z), previous_wrist_pos)
            wrist_relative_landmarks = self.calculate_wrist_relative_landmarks(frame["landmarks"])

            processed_sequence.append({
                "frame_index": frame["frame_index"],
                "timestamp": frame["timestamp"],
                "handedness": frame["handedness"],
                "wrist_displacement": wrist_displacement,
                "landmarks": frame["landmarks"],                #original landmarks
                "relative_landmarks": wrist_relative_landmarks  #coordinates after being subtracted by the wrist position.
            })

            previous_wrist_pos = (wrist_x, wrist_y, wrist_x)
  
        # Trim/pad as needed.
        if len(processed_sequence) < self.sequence_length:
            processed_sequence = self.pad_sequence(processed_sequence)

        elif len(processed_sequence) > self.sequence_length:
            processed_sequence = self.trim_sequence(processed_sequence)

        return processed_sequence
    

    def preprocess_and_save(self):
        '''
        Takes the initalized class object, intakes the data from the source argument.\n
        The data is then processed to include wrist displacement tracking and is padded or trimmed as needed.\n
        Then the data will be saved into the file specified by the source argument.
    
        Arguments:
            Self: PreprocessGestureData

        Return:
            VOID. No return. However, will create a new file containing information from the source that has been processed.</p>
        
        '''
        data = self.load_data()
        preprocessed_sequences = []

        for sequence in data:
            preprocessed_sequence = self.process_sequence(sequence["sequence_data"])
            preprocessed_sequences.append({
                "index": sequence["index"],
                "gesture": sequence["gesture"],
                "sequence_data": preprocessed_sequence
            })

        with open(self.destination, 'w') as file:
            json.dump(preprocessed_sequences, file, indent=4)
        print(f"Preprocessed And Successfully Saved To: {self.destination}")

    def construct_landmark_vector(self, landmarks, handedness, previous_wrist_pos):
        """
        Constructs a feature vector for a frame based on landmarks, handedness, and wrist displacement.\n
        This serves use for the actual gesture_training.py file to quickly construct vectors with incoming data.

        Args:
            landmarks (list): Absolute x, y, z coordinates of each landmark (63 values).
            handedness (str): "Right" or "Left" to indicate hand orientation.
            previous_wrist_pos (list): Previous wrist coordinates [x, y, z] for calculating displacement.

        Returns:
            list: The full feature vector for the frame, including absolute landmarks, 
                  relative landmarks, handedness, and wrist displacement.
        """
        # Handedness as a single feature
        handedness_feature = [1 if handedness == "Right" else 0]
        
        # Wrist displacement relative to the previous frame
        wrist_displacement = self.calculate_wrist_displacement(landmarks[:3], previous_wrist_pos)
        
        # Relative landmarks calculated around the wrist
        relative_landmarks = self.calculate_wrist_relative_landmarks(landmarks)
        
        # Combine all features into a single vector
        feature_vector = landmarks + relative_landmarks + handedness_feature + wrist_displacement
        return feature_vector
        

def process_gesture_directory(source_dir, destination_dir, sequence_length):
    """
    Processes all JSON gesture files in the source directory and outputs them to the destination directory.
    
    Args:
        source_dir (str): Directory containing the source JSON files.
        destination_dir (str): Directory to save the processed JSON files.
        sequence_length (int): Number of frames that each sequence should have.
    """
    os.makedirs(destination_dir, exist_ok=True) 

    source_files = os.listdir(source_dir)

    # Loop through each file in the source directory
    for filename in source_files:
        if filename.endswith('.json'):

            # Create a new filename with "preprocessed" added before the file extension
            base_name, ext = os.path.splitext(filename)
            new_filename = f"{base_name}_preprocessed{ext}"

            source_filepath = os.path.join(source_dir, filename)
            destination_filepath = os.path.join(destination_dir, new_filename)

            processor = PreprocessGestureData(source_filepath, destination_filepath, sequence_length)
            processor.preprocess_and_save()

            print(f"Processed {filename} and saved to {destination_filepath}")



if __name__ == "__main__":
    source_dir = 'gesture_data/'
    destination_dir = 'gesture_data_preprocessed/'
    sequence_length = 30  

    process_gesture_directory(source_dir, destination_dir, sequence_length)