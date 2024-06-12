import base64
import glob
import io
import json
import logging
import os
import random

import numpy as np
import rasterio
import runpod.serverless
import torch
from marinext_wrapper import MariNext
from torch.nn import functional as F

logging.basicConfig(level=logging.INFO)

LOGGER = logging.getLogger(__name__)


def seed_all(seed):
    # Pytorch Reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_models(device: str):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print(current_dir)
    models_files = glob.glob(os.path.join(current_dir, "trained_models", "*.pth"))
    print(models_files)
    models_list = []
    for model_file in models_files:
        model = MariNext(in_chans=11, num_classes=15)

        model.to(device)

        # Load model from specific epoch to continue the training or start the evaluation

        logging.info("Loading model files from folder: %s" % model_file)

        checkpoint = torch.load(model_file, map_location=device)
        checkpoint = {
            k.replace("decoder", "decode_head"): v
            for k, v in checkpoint.items()
            if ("proj1" not in k) and ("proj2" not in k)
        }

        model.load_state_dict(checkpoint)

        del checkpoint  # dereference
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        model.eval()
        models_list.append(model)

    return models_list


def handler(job):
    job_input = job["input"]
    enc_bytes = job_input["image"]
    image_bytes = base64.b64decode(enc_bytes)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    models_list = load_models(device)

    with rasterio.open(io.BytesIO(image_bytes)) as src:
        image = src.read()

    with torch.no_grad():
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        image = image.to(device)
        all_predictions = []
        for model in models_list:
            logits = model(image)
            logits = F.upsample(
                input=logits,
                size=(image.shape[2], image.shape[3]),
                mode="bilinear",
            )
            probs = torch.nn.functional.softmax(logits, dim=1)
            predictions = probs.argmax(1)
            all_predictions.append(predictions)

        all_predictions = torch.cat(all_predictions)
        predictions = torch.mode(all_predictions, dim=0, keepdim=True)[0].cpu().numpy()
    print(predictions.shape)
    base_64_prediction = base64.b64encode(predictions.tobytes()).decode("utf-8")
    return json.dumps(
        {
            "prediction": base_64_prediction,
            "shape": predictions.shape,
            "dtype": str(predictions.dtype),
        }
    )


runpod.serverless.start({"handler": handler})
