# Disaster Response Analysis


## Installation

You can rou the project using Docker containers.

There are Dockerfiles inside the notebook/container and app folders.

There are example commands, notebooks\_container\_start.sh.sample and app/run_container.sh.sample files, that shows how to run the respective containers.

To preprocess the dataset and insert it into the database file you could run a comand like bash -c "cd data && python process\_data.py messages.csv categories.csv DisasterResponse.db".

To train the classifier and save it into a pickle file you could run a comand like bash -c "cd models && python train\_classifier.py ../data/DisasterResponse.db best_model.pkl". 

If you are running the app inside the container you could use the **exec** command to run the above commands.

The container of the app will start the server at the localhost at the port 3001.

There are requiriments.txt files toghether with the Dockerfiles, if you want to run the code a different way.

## Dataset

The dataset contains data about disasters and messages people send requesting or offering something.

It also contains some classifications for the messages.

## Project Organization

The app folder has the app's files for the web project. It was based on the Udacity's Data Scientist Nanodegree Program.

The app/dataset folder contains the .csv data used for the analysis and a .db database used to be the output of the preprocessing data step.

The app/models folder contains the machine learning saved models.

The notebooks folder contains jupyter notebooks used to analyze the datasets.

These notebooks were copied from the Udacity Data Scientist Nanodegree Program.

## Acknowledgments

-Udacity, for providing the ideia and the notebooks for the project.

-Figure Eight Inc. for providing the dataset.
