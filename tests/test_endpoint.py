import base64
import io
import json

import numpy as np
import pytest
import rasterio
from runpod import Endpoint

from tests.conftest import ENDPOINT_ID

BANDS = """Band 2 (Blue) - 490 nm
Band 3 (Green) - 560 nm
Band 4 (Red) - 665 nm
Band 5 (Red Edge 1) - 705 nm
Band 6 (Red Edge 2) - 740 nm
Band 7 (Red Edge 3) - 783 nm
Band 8 (NIR) - 842 nm
Band 8A (Narrow NIR) - 865 nm
Band 11 (SWIR 1) - 1610 nm
Band 12 (SWIR 2) - 2190 nm
Band 1 (Coastal aerosol) - 443 nm ."""


@pytest.fixture
def input_data():
    with open("tests/data/Scene_1_L2R_rhorc_6.tif", "rb") as f:
        return f.read()


def test_invoke(input_data: bytes):
    with rasterio.open(io.BytesIO(input_data)) as src:
        meta = src.meta.copy()

    # with rasterio.open(io.BytesIO(expected_prediction)) as src:
    #     expected_numpy = src.read()

    encoded_input = base64.b64encode(input_data).decode("utf-8")
    endpoint = Endpoint(ENDPOINT_ID)
    run_request = endpoint.run_sync(
        request_input={"input": {"image": encoded_input}},
        timeout=60,
    )
    request = json.loads(run_request)
    dtype = request["dtype"]
    shape = request["shape"]
    print(shape)
    print(dtype)
    np_buffer = base64.b64decode(request["prediction"])
    prediction = np.frombuffer(np_buffer, dtype=dtype).reshape(
        shape[0], shape[1], shape[2]
    )
    meta.update(
        dtype="float32",
        count=1,
    )
    with rasterio.open("tests/data/response_prediction.tiff", "w+", **meta) as dst:
        dst.write(prediction)

    # np.testing.assert_array_almost_equal(prediction, expected_numpy, decimal=3)
