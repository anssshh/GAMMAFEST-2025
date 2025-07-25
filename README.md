# GAMMAFEST-2025
GammaFest 2025 challenges participants to build a machine learning model that predicts citation links between scientific papers. Using full texts, metadata, and topic labels, the goal is to help the Elbaf Library recommend relevant research. Models are evaluated using the Matthews Correlation Coefficient (MCC).

# Competition Overview
## Background
For centuries, the Elbaf Library has served as the guardian of the world's knowledge. From ancient wisdom to modern science, everything is meticulously recorded in thousands of papers stored within its walls. The library is watched over by Biblo, a faithful guardian who ensures that knowledge remains accessible to those in need. However, as time has passed, the library’s reference management system has begun to show various weaknesses, hindering access to the vast knowledge it holds.

Now, a group of Holy Knights has arrived, seeking to document their intellectual journeys in the form of scientific papers. They hope to write works that will become a legacy for future generations, but they face a major challenge—how can they find the right references to support their research? They turn to Biblo for help, but Biblo no longer has the strength to assist the Holy Knights in finding, recommending, or connecting relevant papers.

## Problem
The current paper recommendation systems are not effective enough in providing relevant literature suggestions for the Holy Knights. The Elbaf Library needs a system that can help manage and present appropriate references for various research needs. Moreover, mapping relationships between papers remains a complex task, resulting in important references often being overlooked.

Participants in this competition are asked to build a machine learning model to solve this challenge. The model should be able to predict citation relationships between papers. The core issue is to help the Elbaf Library build an optimal reference recommendation system.

To evaluate the models, the Matthews Correlation Coefficient (MCC) will be used as the evaluation metric.

## Dataset Description
The dataset consists of pairs of scientific documents, each equipped with metadata such as titles, abstracts, and publication years. Each row in the dataset represents the possibility that the first document (paper) cites the second document (referenced_paper).

Participants are tasked with building a machine learning model to predict the value of is_referenced, indicating whether a citation relationship actually exists based on the available information.

Files:
- Folder Paper Database: Contains the full texts of documents in .txt format, with filenames corresponding to paper_id.
- papers_metadata.csv: Contains complete metadata for each document.
- train.csv: Contains document pairs with labels indicating whether the paper cites the referenced_paper.
- test.csv: Contains document pairs without labels, to be predicted by participants.
- sample_submission.csv: Sample format for submitting predictions on the test data.

## Evaluation
Submissions are evaluated using .csv files with the specified column headers (see the sample submission). The MCC (Matthews Correlation Coefficient) metric will be used to assess the model’s performance. A portion of the test dataset results will form the public leaderboard. The full results will be revealed in the private leaderboard after the competition ends.

# Competition Link
https://www.kaggle.com/competitions/gammafest25
