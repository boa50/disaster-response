# Disaster Response Analysis


## Installation

You can start the notebooks container with a command like: docker run --rm -p 8888:8888 -e JUPYTER_ENABLE_LAB=yes -v "${PWD}":/home/jovyan/work IMAGE_NAME

## Dataset

The dataset contains data about disasters and messages people send requesting or offering something.

It also contains some classifications for the messages.

## Project Organization

The dataset folder contains the .csv data used for the analysis and a .db database used to be the output of the preprocessing data step.

The models folder contains the machine learning saved models.

The app folder has the app's files for the web project. It was based on the Udacity's Data Scientist Nanodegree Program.

The notebooks folder contains jupyter notebooks used to analyze the datasets.

These notebooks were copied from the Udacity Data Scientist Nanodegree Program.

The notebooks folder has a container folder too.

The container folder contains the Dockerfile and the requirements.txt used to start the container.

## Acknowledgments

-Udacity, for providing the ideia and the notebooks for the project.

-Figure Eight Inc. for providing the dataset.
