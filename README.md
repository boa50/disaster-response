# Disaster Response Analysis

This project tries to analyze data of text messages about real life disasters.

The idea is to filter the messages according to some categories like: food, water, etc.

This is important because, during a disaster, a lot of messages are sent and many of them are not very useful, so we tried to take the meaning of the messages based on a Machine Learning algorithm.

Some charts were presented to try to identify the categories with more necessity of help.

The messages show that, during a disaster, much more messages about requests are received then messages of people offering help. This could be a problem, because sometimes is very difficult to connect these people to help each other.

This dataset has a problem about the imbalanced values of the categories, making difficlut to classify some of the categories.

Because of this the F1 score was chosen to be the most important metric for the algorithm, to make the right decisions when the categories are true more important. If we chose the accuracy to be the most important metric, the model could say that all categories are false for all sentences and still has a good accuracy.

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

The messages and the classifications of the messages are divided into two .csv files (messages.csv and categories.csv)

## Project Organization

The app folder has the app's files for the web project. It was based on the Udacity's Data Scientist Nanodegree Program.

The app/dataset folder contains the .csv data used for the analysis and a .db database used to be the output of the preprocessing data step.

The app/models folder contains the machine learning saved models.

The notebooks folder contains jupyter notebooks used to analyze the datasets.

These notebooks were copied from the Udacity Data Scientist Nanodegree Program.

## Acknowledgments

-Udacity, for providing the ideia and the notebooks for the project.

-Figure Eight Inc. for providing the dataset.
