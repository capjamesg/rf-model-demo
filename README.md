# Roboflow Chess Model (Example)

This project queries a pretrained chess object detection model hosted on [Roboflow](https://roboflow.com). The project:

1. Prints out all of the predictions over a given confidence threshold.
2. Saves the annotated predictions to a file.

## Getting Started

To get started with this project, first set up a virtual environment in which you can install the required dependencies:

    python3 -m venv venv
    source venv/bin/activate
    pip3 install -r requirements.txt

Next, copy the `example.config.py` file into a file called `config.py`:

    cp example.config.py config.py

Open up the `config.py` file and update its contents with your:

1. Roboflow model name
2. API key
3. Project version name
4. The name of the image on which you want to run a prediction
5. The confidence threshold above which predictions are displayed to the console

Now you are ready to run the project. You can do so using this command:

    python3 model.py

You will see an output with the confidence thresholds retrieved from the Roboflow API, like this:

    

## Technologies

This project was built using Python and Roboflow.

## Contributors

- capjamesg