import tensorflow.keras as keras
import numpy as np
import librosa

MODEL_PATH = "model.h5"
NUM_SAMPLES_TO_CONSIDER = 22050 # 1 sec sound

class _Keyword_Recognition_Service:
    model = None
    _mappings = [
        "right",
        "go",
        "no",
        "left",
        "stop",
        "up",
        "down",
        "yes",
        "on",
        "off"
    ]
    _instance = None

    def predict(self, file_path):

        # extract the MFCCs
        MFCCs = self.preprocess(file_path) # (# segment, # coefficients)

        # Convert 2d MFCCs array into 4D array -> (#samples, # segments, # coefficients, # channels)
        MFCCs = MFCCs[np.newaxis, ..., np.newaxis]  # Adding new dimensions #samples and # channels

        # Make prediction
        predictions = self.model.predict(MFCCs)  # [[ 0.1, 0.6, 0.1, ..... ]]
        predicted_index = np.argmax(predictions) # Maximum of the above indices
        predicted_keyword = self._mappings[predicted_index] # Mapping the keyword with the index

        return predicted_keyword

    def preprocess(self, file_path, n_mfcc = 13, hop_length = 512, n_fft = 2048):

        # load audio file
        signal, sr = librosa.load(file_path)

        # ensure consistency in the audio file length
        if len(signal) > NUM_SAMPLES_TO_CONSIDER:
            signal = signal[:NUM_SAMPLES_TO_CONSIDER] # Resizing the signal

        # extract MFCCs
        MFCCs = librosa.feature.mfcc(signal, n_mfcc = n_mfcc, hop_length = hop_length, n_fft = n_fft)

        return MFCCs.T

def Keyword_Recognition_Service():

    # Ensure that we only have one instance of KRS
    if _Keyword_Recognition_Service._instance is None:
        _Keyword_Recognition_Service._instance = _Keyword_Recognition_Service()
        _Keyword_Recognition_Service.model = keras.models.load_model(MODEL_PATH)

    return _Keyword_Recognition_Service._instance

if __name__ == "__main__":

    krs = Keyword_Recognition_Service()
    keyword1 = krs.predict("Test/go.wav")
    keyword2 = krs.predict("Test/off.wav")
    keyword3 = krs.predict("Test/stop.wav")

    print(f"Predicted keywords: {keyword1}, {keyword2}, {keyword3}")
