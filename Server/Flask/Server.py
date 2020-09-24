from flask import Flask, request, jsonify
import random
from keyword_recognition_service import Keyword_Recognition_Service
import os
app = Flask(__name__)

'''
When we get a POST request in flask we need to process that request and send a Json file back
'''
@app.route("/predict", methods=["POST"])

def predict():

    # Get the audio file and save it
    audio_file = request.files["file"]
    file_name = str(random.randint(0,100000))
    audio_file.save(file_name)

    # Invoke Keyword Recognition Service
    krs =  Keyword_Recognition_Service()

    # Make prediction
    predicted_keyword = krs.predict(file_name)

    # Remove the audio file
    os.remove(file_name)

    # Send back the predicted keyword in Json format
    data = {"keyword": predicted_keyword}

    return jsonify(data)

if __name__ == "__main__":
    app.run(debug=False)
