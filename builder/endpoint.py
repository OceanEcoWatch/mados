import os

import runpod


def update_endpoint(image_name: str, endpoint_id: str):
    runpod.api_key = os.environ["RUNPOD_API_KEY"]

    new_template = runpod.create_template(
        name=image_name,
        image_name=image_name,
        is_serverless=True,
    )
    print(f"Created template: {new_template}")
    updated_endpoint = runpod.update_endpoint_template(
        endpoint_id=endpoint_id,
        template_id=new_template["id"],
    )
    print(f"Updated endpoint: {updated_endpoint}")


if __name__ == "__main__":
    update_endpoint("marciejj/plastic_detection_model:1.0.1", "i1dp5odzq2kbgc")
