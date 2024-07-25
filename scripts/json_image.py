import base64
import json

# Open the image file in binary mode
with open("tests/data/Scene_0_L2R_rhorc_7.tif", "rb") as image_file:
    # Read the image file and encode it in Base64
    encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

# Print the Base64 string (for verification)`Ω``


# Create a dictionary to hold the JSON structure
json_data = {"input": {"image": encoded_string}}


# Convert the dictionary to a JSON string
json_string = json.dumps(json_data)

# Print the JSON string (for verification)
with open("tests/data/test_input.json", "w") as json_file:
    json_file.write(json_string)
