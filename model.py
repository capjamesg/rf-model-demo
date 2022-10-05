from typing import Tuple

from roboflow import Roboflow
from config import PROJECT_NAME, VERSION, IMAGE_NAME, CONFIDENCE_THRESHOLD, API_KEY


def initialise_model() -> Tuple[dict, Roboflow]:
    """
    Initialise the model for the provided version then run a prediction on a given image.
    """
    rf = Roboflow(api_key=API_KEY)

    project = rf.workspace().project(PROJECT_NAME)

    model = project.version(VERSION).model

    # infer on a local image
    predictions = model.predict(
        IMAGE_NAME, confidence=CONFIDENCE_THRESHOLD, overlap=30
    ).json()

    return predictions, model


def show_predictions(predictions: dict) -> None:
    """
    Print all of the predictions made by the model hosted on Roboflow to the console.
    """
    result_count = 0

    for item in predictions["predictions"]:
        # round to two decimal places
        confidence = round(item["confidence"] * 100, 2)

        print(
            "Found {} with confidence of {} (x = {}, y = {})".format(
                item["class"], str(confidence), item["x"], item["y"]
            )
        )
        result_count += 1

    print(
        "Found {} items in total above {}% confidence threshold.".format(
            result_count, CONFIDENCE_THRESHOLD
        )
    )


if __name__ == "__main__":
    predictions, model = initialise_model()

    show_predictions(predictions)

    prediction_file = "prediction.jpg"

    # save annotated prediction to file
    model.predict(IMAGE_NAME, confidence=CONFIDENCE_THRESHOLD, overlap=30).save(
        prediction_file
    )

    print("Saved visual prediction to {}.".format(prediction_file))
