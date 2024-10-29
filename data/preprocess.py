import os
import json



class PreprocessGestureData:
    '''

    <h2>Class Definition: PreprocessGestureData</h2>

    This is a class designed for preprocessing data with far more ease when it comes to the gesture data.\n

    This is made in case there is a need to work with new data/information in future cases.
    
    <pre>PreprocessGestureData(self, destination, source, sequence_length)</pre>
    
        <strong>Arguments:</strong>
        <ul>
            <li><strong>source:</strong> <code>PATH</code>, file path of the JSON file to read information from.</li>
            <li><strong>destination:</strong> <code>PATH</code>, file path of the JSON file (new or not) to deposit all the information into.</li>
            <li><strong>sequence_length:</strong> <code>int</code>, Number of frames that each sequence should take up.</li>
        </ul>

    <p>If the <code>sequence_length</code> of the frames is smaller than from the data set, we will choose the center frames and cut out the extreme ends.\n
    If the <code>sequence_length</code> of the frames is larger than from the data set, we will duplicate the extreme ends to fill up both ends equally until the target is reached.</p>

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
    
    
    def translate_landmarks_around_wrists(self, frame):
        '''
        <p>Given a frame from the JSON file that has been made from data gathering,\n
        grab every landmark and normalize them around landmark 0 (wrist).</p>
        
        <pre>translate_landmarks_around_wrists(self, frame)</pre>
    
        <strong>Arguments:</strong>
        <ul>
            <li>frame, Frame of a sequence, includes index #, timestamp, handedness of the hand, and landmark coordinates</li>
        </ul>

        <strong>Return:</strong>\n
        Array of landmarks where every landmark is normalized/centered around the wrist.</p>

        '''

        wrist_x, wrist_y, wrist_z = frame["landmarks"][:3]
        translated_landmarks = []

        for i in range(0, len(frame["landmarks"]), 3):
            x, y, z = frame["landmarks"][i:i+3]
            translated_landmarks.extend([
                x - wrist_x,
                y - wrist_y,
                z - wrist_z
            ])

        return translated_landmarks
    
    def calculate_wrist_displacement(self, current_wrist, prior_wrist):
        '''
        <p>Given a set of coordinates for the current wrists, and the prior wrists,\n
        will calcualte the displacement by subtracting the prior from the current position.</p>
        
        <pre>calculate_wrist_displacement(self, current_wrist, prior_wrist)</pre>
    
        <strong>Arguments:</strong>
        <ul>
            <li>current_wrist, array of 3 integers represnting the x,y,z coordinates of the current wrist</li>
            <li>prior_wrist, array of 3 integers represnting the x,y,z coordinates of the prior wrist</li>
        </ul>

        <strong>Return:</strong>\n
        Array of 3 int, representing the difference in current by prior.</p>

        '''
        if prior_wrist is None:
            return [0, 0, 0]
        else:
            return [
                current_wrist[0] - prior_wrist[0],
                current_wrist[1] - prior_wrist[1],
                current_wrist[2] - prior_wrist[2]
            ]
    
    def pad_sequence(self, sequence):
        '''
        <p>Given a sequence from the data_gathered JSON file that has less frames than desired, pad the sequence.\n
        Identifies the amount of frames needed, and then splits the amount between the front and end of the sequence,\n
        where the front and end frames are duplicated for their respective amount.</p>
        
        <pre>pad_sequence(self, sequence)</pre>
    
        <strong>Arguments:</strong>
        <ul>
            <li>Sequence: List of frames containing information of the hand being tracked.</li>
        </ul>

        <strong>Return:</strong>\n
        A padded sequence that has either end duplicated to retain natural flow and achieving the desired frame count.</p>
        
        '''
    
        num_frames_needed = self.sequence_length - len(sequence)

        start_count = num_frames_needed // 2
        end_count = num_frames_needed - start_count

        padded_sequence = [sequence[0]] * start_count + sequence + [sequence[-1]] * end_count 
        
        return padded_sequence
    
    def trim_sequence(self, sequence):
        '''
        <p>Given a sequence from the data_gathered JSON file that has less frames than desired, trim the sequence.\n
        Identifies the amount of frames needed to remove, and then splits the amount between the front and end of the sequence,\n
        where the front and end frames are removed for their respective amount.</p>
        
        <pre>trim_sequence(self, sequence)</pre>
    
        <strong>Arguments:</strong>
        <ul>
            <li>Sequence: List of frames containing information of the hand being tracked.</li>
        </ul>

        <strong>Return:</strong>\n
        A trimmed sequence that has either end removed to retain natural flow and achieving the desired frame count.</p>
        
        '''
        excess_frame_count = len(sequence) - self.sequence_length

        start_count = excess_frame_count // 2
        end_count = excess_frame_count - start_count

        trimmed_sequence = sequence[start_count:len(sequence) - end_count]
        
        return trimmed_sequence
    

    def process_sequence(self, sequence_data):
        '''
        <p>Given a sequence from the data_gathered JSON file that has less frames than desired, process the sequence.\n
        Will normalize all the landmarks present in the sequence based on the frame's wrist position.\n
        If needed, the sequence will be either trimmed or padded to maintain constant frame count across the data.\n</p>
        
        <pre>process_sequence(self, sequence)</pre>
    
        <strong>Arguments:</strong>
        <ul>
            <li>Sequence: List of frames containing information of the hand being tracked.</li>
        </ul>

        <strong>Return:</strong>\n
        A sequence processed to have landmarks normalized about the wrist and the proper frame count.</p>
        
        '''

        processed_sequence = []
        previous_wrist_pos = None

        # For each frame in the sequence, normalize/translate the landmarks
        # Then identify the wrist displacement.
        # Add the new/updated info to the processed array.
        for frame in sequence_data:
            wrist_x, wrist_y, wrist_x = frame["landmarks"][:3]
            normalized_landmarks = self.translate_landmarks_around_wrists(frame)
            wrist_displacement = self.calculate_wrist_displacement((wrist_x, wrist_y, wrist_x), previous_wrist_pos)

            processed_sequence.append({
                "frame_index": frame["frame_index"],
                "timestamp": frame["timestamp"],
                "handedness": frame["handedness"],
                "wrist_displacement": wrist_displacement,
                "landmarks": normalized_landmarks
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
        <p>Takes the initalized class object, intakes the data from the <code>source</code> argument.\n
        The data is then processed, having every landmark coordinate translate to center around the wrist.\n
        The sequences have their frames padded/trimmed as needed.\n
        Then the data will be saved into the file specified by the <code>source</code> argument.</p>
        
        <pre>preprocess_and_save(self, sequence)</pre>
    
        <strong>Arguments:</strong>
        <ul>
            <li>Self: PreprocessGestureData</li>
        </ul>

        <strong>Return:</strong>\n
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










    

        