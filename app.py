from flask import Flask, request
import json
import os

from predict import predict

app = Flask(__name__)


@app.route("/")
def hello():

	return json.dumps({"message" : "Hello World", "statusCode":200})

@app.route("/predict", methods=["GET"])
def calcaulate_sentiment():
	text = request.args.get("text")
	prediction = predict(text)

	return json.dumps({"message" : f"This is a {prediction} sentimental tweet", "statusCode":200})



if __name__ == "__main__":

	port = int(os.environ.get("PORT", 5000))

	app.run(host="0.0.0.0", port=port)

