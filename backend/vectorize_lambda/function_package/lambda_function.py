import boto3
import os
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import hf_hub_download
import transformers
from huggingface_hub import snapshot_download
from __future__ import unicode_literals
import re
import unicodedata
import scipy
import matplotlib.pyplot as plt
from PIL import Image
import math



class ClipTextModel(nn.Module):
    def __init__(self, model_name_or_path, device=None):
        super(ClipTextModel, self).__init__()

        if os.path.exists(model_name_or_path):
            # load from file system
            output_linear_state_dict = torch.load(os.path.join(model_name_or_path, "output_linear.bin"), map_location=device)
        else:
            # download from the Hugging Face model hub
            filename = hf_hub_download(repo_id=model_name_or_path, filename="output_linear.bin")
            output_linear_state_dict = torch.load(filename)

        self.model = AutoModel.from_pretrained(model_name_or_path)
        config = self.model.config

        self.max_cls_depth = 6

        sentence_vector_size = output_linear_state_dict["bias"].shape[0]
        self.sentence_vector_size = sentence_vector_size
        self.output_linear = nn.Linear(self.max_cls_depth * config.hidden_size, sentence_vector_size)
        self.output_linear.load_state_dict(output_linear_state_dict)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, 
                                                       is_fast=True, do_lower_case=True)

        self.eval()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.to(self.device)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
    ):
        output_states = self.model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=True,
            return_dict=True,
        )
        token_embeddings = output_states[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        hidden_states = output_states["hidden_states"]

        output_vectors = []

        # cls tokens
        for i in range(1, self.max_cls_depth + 1):
            cls_token = hidden_states[-1 * i][:, 0]
            output_vectors.append(cls_token)

        output_vector = torch.cat(output_vectors, dim=1)
        logits = self.output_linear(output_vector)

        output = (logits,) + output_states[2:]
        return output

    @torch.no_grad()
    def encode_text(self, texts, batch_size=8, max_length=64):
        model.eval()
        all_embeddings = []
        iterator = range(0, len(texts), batch_size)
        for batch_idx in iterator:
            batch = texts[batch_idx:batch_idx + batch_size]

            encoded_input = self.tokenizer.batch_encode_plus(
                batch, max_length=max_length, padding="longest", 
                truncation=True, return_tensors="pt").to(self.device)
            model_output = self(**encoded_input)
            text_embeddings = model_output[0].cpu()

            all_embeddings.extend(text_embeddings)

        # return torch.stack(all_embeddings).numpy()
        return torch.stack(all_embeddings)        

    def save(self, output_dir):
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        torch.save(self.output_linear.state_dict(), os.path.join(output_dir, "output_linear.bin"))

class ClipVisionModel(nn.Module):
    def __init__(self, model_name_or_path, device=None):
        super(ClipVisionModel, self).__init__()

        if os.path.exists(model_name_or_path):
            # load from file system
            visual_projection_state_dict = torch.load(os.path.join(model_name_or_path, "visual_projection.bin"))
        else:
            # download from the Hugging Face model hub
            filename = hf_hub_download(repo_id=model_name_or_path, filename="visual_projection.bin")
            visual_projection_state_dict = torch.load(filename)

        self.model = transformers.CLIPVisionModel.from_pretrained(model_name_or_path)
        config = self.model.config

        self.feature_extractor = transformers.CLIPFeatureExtractor.from_pretrained(model_name_or_path)

        vision_embed_dim = config.hidden_size
        projection_dim = 512

        self.visual_projection = nn.Linear(vision_embed_dim, projection_dim, bias=False)
        self.visual_projection.load_state_dict(visual_projection_state_dict)

        self.eval()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.to(self.device)

    def forward(
        self,
        pixel_values=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_states = self.model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        image_embeds = self.visual_projection(output_states[1])

        return image_embeds

    @torch.no_grad()
    def encode_image(self, images, batch_size=8):
        model.eval()
        all_embeddings = []
        iterator = range(0, len(images), batch_size)
        for batch_idx in iterator:
            batch = images[batch_idx:batch_idx + batch_size]

            encoded_input = self.feature_extractor(batch, return_tensors="pt").to(self.device)
            model_output = self(**encoded_input)
            image_embeddings = model_output.cpu()

            all_embeddings.extend(image_embeddings)

        # return torch.stack(all_embeddings).numpy()
        return torch.stack(all_embeddings)        

    @staticmethod
    def remove_alpha_channel(image):
        image.convert("RGBA")
        alpha = image.convert('RGBA').split()[-1]
        background = Image.new("RGBA", image.size, (255, 255, 255))
        background.paste(image, mask=alpha)
        image = background.convert("RGB")
        return image

    def save(self, output_dir):
        self.model.save_pretrained(output_dir)
        self.feature_extractor.save_pretrained(output_dir)
        torch.save(self.visual_projection.state_dict(), os.path.join(output_dir, "visual_projection.bin"))

class ClipModel(nn.Module):
    def __init__(self, model_name_or_path, device=None):
        super(ClipModel, self).__init__()

        if os.path.exists(model_name_or_path):
            # load from file system
            repo_dir = model_name_or_path
        else:
            # download from the Hugging Face model hub
            repo_dir = snapshot_download(model_name_or_path)

        self.text_model = ClipTextModel(repo_dir, device=device)
        self.vision_model = ClipVisionModel(os.path.join(repo_dir, "vision_model"), device=device)

        with torch.no_grad():
            logit_scale = nn.Parameter(torch.ones([]) * 2.6592)
            logit_scale.set_(torch.load(os.path.join(repo_dir, "logit_scale.bin"), map_location=device).clone().cpu())
            self.logit_scale = logit_scale

        self.eval()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.to(self.device)

    def forward(self, pixel_values, input_ids, attention_mask, token_type_ids):
        image_features = self.vision_model(pixel_values=pixel_values)
        text_features = self.text_model(input_ids=input_ids, 
                                        attention_mask=attention_mask, 
                                        token_type_ids=token_type_ids)[0]

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        return logits_per_image, logits_per_text

    @torch.no_grad()
    def encode(self, images, texts, batch_size=8, max_length=64):
        model.eval()
        image_features = self.vision_model.encode_image(images, batch_size=batch_size)
        text_features = self.text_model.encode_text(texts, batch_size=batch_size, max_length=max_length)

        image_features = image_features.to(self.device)
        text_features = text_features.to(self.device)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        logits_per_image = logits_per_image.cpu()
        logits_per_text = logits_per_text.cpu()

        return logits_per_image, logits_per_text

    def save(self, output_dir):
        torch.save(self.logit_scale, os.path.join(output_dir, "logit_scale.bin"))
        self.text_model.save(output_dir)
        self.vision_model.save(os.path.join(output_dir, "vision_model"))

# https://github.com/neologd/mecab-ipadic-neologd/wiki/Regexp.ja から引用・一部改変

def unicode_normalize(cls, s):
    pt = re.compile('([{}]+)'.format(cls))

    def norm(c):
        return unicodedata.normalize('NFKC', c) if pt.match(c) else c

    s = ''.join(norm(x) for x in re.split(pt, s))
    s = re.sub('－', '-', s)
    return s

def remove_extra_spaces(s):
    s = re.sub('[ 　]+', ' ', s)
    blocks = ''.join(('\u4E00-\u9FFF',  # CJK UNIFIED IDEOGRAPHS
                      '\u3040-\u309F',  # HIRAGANA
                      '\u30A0-\u30FF',  # KATAKANA
                      '\u3000-\u303F',  # CJK SYMBOLS AND PUNCTUATION
                      '\uFF00-\uFFEF'   # HALFWIDTH AND FULLWIDTH FORMS
                      ))
    basic_latin = '\u0000-\u007F'

    def remove_space_between(cls1, cls2, s):
        p = re.compile('([{}]) ([{}])'.format(cls1, cls2))
        while p.search(s):
            s = p.sub(r'\1\2', s)
        return s

    s = remove_space_between(blocks, blocks, s)
    s = remove_space_between(blocks, basic_latin, s)
    s = remove_space_between(basic_latin, blocks, s)
    return s

def normalize_neologd(s):
    s = s.strip()
    s = unicode_normalize('０-９Ａ-Ｚａ-ｚ｡-ﾟ', s)

    def maketrans(f, t):
        return {ord(x): ord(y) for x, y in zip(f, t)}

    s = re.sub('[˗֊‐‑‒–⁃⁻₋−]+', '-', s)  # normalize hyphens
    s = re.sub('[﹣－ｰ—―─━ー]+', 'ー', s)  # normalize choonpus
    s = re.sub('[~∼∾〜〰～]+', '〜', s)  # normalize tildes (modified by Isao Sonobe)
    s = s.translate(
        maketrans('!"#$%&\'()*+,-./:;<=>?@[¥]^_`{|}~｡､･｢｣',
              '！”＃＄％＆’（）＊＋，－．／：；＜＝＞？＠［￥］＾＿｀｛｜｝〜。、・「」'))

    s = remove_extra_spaces(s)
    s = unicode_normalize('！”＃＄％＆’（）＊＋，－．／：；＜＞？＠［￥］＾＿｀｛｜｝〜', s)  # keep ＝,・,「,」
    s = re.sub('[’]', '\'', s)
    s = re.sub('[”]', '"', s)
    s = s.lower()
    return s

def normalize_text(text):
    return normalize_neologd(text)

def search_image(query_vector, target_vectors, target_images, closest_n=3):
    distances = scipy.spatial.distance.cdist(
        query_vector, target_vectors, metric="cosine"
    )[0]

    results = zip(range(len(distances)), distances)
    results = sorted(results, key=lambda x: x[1])

    # タイリング表示
    plt.figure(dpi=140, figsize=(6,6))

    for i, (idx, distance) in enumerate(results[0:closest_n]):
        image = target_images[idx]

        sp = plt.subplot(math.ceil(closest_n / 4), 4, i + 1)
        plt.imshow(image)
        text = sp.text(-32, 0, f"{i + 1}: {distance:0.5f}", ha="left", va="bottom", color="black", fontsize=12)
        plt.axis("off")    

def search_image_by_text(text, target_vectors, target_images, closest_n=3):
    text = normalize_text(text)
    text_features = model.text_model.encode_text([text]).numpy()
    search_image(text_features, target_vectors, target_images, closest_n)

def search_image_by_image(image, target_vectors, target_images, closest_n=3):
    image_features = model.vision_model.encode_image([image]).numpy()
    search_image(image_features, target_vectors, target_images, closest_n)

s3_client = boto3.client('s3')

def lambda_handler(event, context):
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']
    download_path = '/tmp/{}{}'.format(uuid.uuid4(), key)
    
    s3_client.download_file(bucket, key, download_path)
    
    with open(download_path, 'r') as file:
        content = file.read()
        print(content)
    
device = "cuda" if torch.cuda.is_available() else "cpu"
model = ClipModel("sonoisa/clip-vit-b-32-japanese-v1", device=device)