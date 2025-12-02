import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import agent

app = Flask(__name__)
CORS(app)

PORT = int(os.environ.get("PORT", 10000))

MODEL_PATH = "./model/health_advice_model_best.ckpt"
TOKENIZER_PATH = "./model/health_tokenizer.json"
CONFIG_PATH = "./model/config.json"

print("正在加载模型，请稍候...")
MODEL, TOKENIZER, CONFIG = agent.load_model_for_inference(
    MODEL_PATH,
    TOKENIZER_PATH,
    CONFIG_PATH
)
print("模型加载完成！")

@app.route("/api/advice", methods=["POST"])
def advice():
    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "消息格式错误"}), 400
    text = data["message"]
    try:
        reply = agent.generate_advice(MODEL, TOKENIZER, text, CONFIG)
        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def home():
    return "Health Helper AI Backend (Railway) is running."

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
