import base64
import glob
import io
import json
import logging
import os

import rasterio
import runpod.serverless
import torch
from marinext_wrapper import MariNext

logging.basicConfig(level=logging.INFO)

LOGGER = logging.getLogger(__name__)


def handler(job):
    job_input = job["input"]
    enc_bytes = job_input["image"]
    image_bytes = base64.b64decode(enc_bytes)

    with rasterio.open(io.BytesIO(image_bytes)) as src:
        image = src.read()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    current_dir = os.path.dirname(os.path.realpath(__file__))
    models_files = glob.glob(os.path.join(current_dir, "trained_models", "*.pth"))
    models_list = []
    for model_file in models_files:
        model = MariNext(in_chans=12, num_classes=15)

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

    prediction = predict(model, image=image, device=device)

    base_64_prediction = base64.b64encode(prediction.tobytes()).decode("utf-8")
    return json.dumps({"prediction": base_64_prediction})


runpod.serverless.start({"handler": handler})
