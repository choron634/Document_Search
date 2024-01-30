import torch
from clip import clip

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("sonoisa/clip-vit-b-32-japanese-v1", device=device)

def model_fn(model_dir):
    return model

def input_fn(request_body, request_content_type):
    # リクエストデータの前処理を実装します。
    # ここでは、request_bodyをそのまま返す簡単な例を示します。
    return request_body

def predict_fn(input_data, model):
    if 'image' in input_data:
        image = preprocess(input_data['image']).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image)
        return image_features
    elif 'text' in input_data:
        text = input_data['text']
        with torch.no_grad():
            text_features = model.encode_text(clip.tokenize([text]).to(device))
        return text_features
    else:
        raise ValueError("Invalid input data. It should contain either 'image' or 'text'.")

def output_fn(prediction, response_content_type):
    # 予測結果の後処理を実装します。
    # ここでは、predictionをそのまま返す簡単な例を示します。
    return prediction