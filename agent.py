# Placeholder agent.py
def load_model_for_inference(ckpt_path, tokenizer_path, config_path):
    model = {"ckpt": ckpt_path}
    tokenizer = {"file": tokenizer_path}
    config = {"file": config_path}
    return model, tokenizer, config

def generate_advice(model, tokenizer, text, config):
    return "（示例回答）感谢你的提问：'{}'。这是一个示范回复，请替换为真实模型输出。".format(text)
