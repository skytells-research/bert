from flask import Flask,request,jsonify
from flask_cors import CORS

from bert import QA

app = Flask(__name__)
CORS(app)
lang = request.json["lang"]
if (lang == "ar") {
    modelName = "bert-base-multilingual-cased"
}else{
    modelName = "bert-base-multilingual-cased"
}
model = QA(modelName)

@app.route("/predict",methods=['POST'])
def predict():
    doc = request.json["document"]
    q = request.json["question"]
    try:
        out = model.predict(doc,q)
        return jsonify({"result":out})
    except Exception as e:
        print(e)
        return jsonify({"result":"Model Failed"})

if __name__ == "__main__":
    app.run('localhost',port=8000)
