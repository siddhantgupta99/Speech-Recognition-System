import requests
URL = "http://18.204.9.14/predict"
TEST_AUDIO_FILE_PATH = "Test/stop.wav"

if __name__ == "__main__":

    audio_file = open(TEST_AUDIO_FILE_PATH, "rb")
    values = {"file": (TEST_AUDIO_FILE_PATH, audio_file, "audio/wav")}
    response = requests.post(URL, files = values)
    data = response.json()

    print(f"Predicted keyword is: {data['keyword']}")