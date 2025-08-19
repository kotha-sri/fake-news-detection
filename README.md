# Fake News Detection
<img src="static/fake-news-logo" alt="logo" width="200"/>


Final project for the Deep Dive Into AI summer program at the University of Texas at Dallas.

The [WELFake Dataset](https://data.niaid.nih.gov/resources?id=zenodo_4561252) was used to train the model, made by combining the four most popular datasets (Kaggle, McIntire, Reuters, BuzzFeed Political).

The model uses a logistic regression to determine the validity of a website, with 95% accuracy. It is optimized using a C-value of 20, solver of liblinear, and l2 regularization. The model is adapted into a website where a website url is inputted and the results of the model subsequently displayed.

Note: other machine learning systems would likely provide better results (Transformer) but the resources available to us didn't allow for traning more complicated models. 

Created by Chi Ming Fung (frontend), Soumith Reddy (model training and optimization), and Srikar Kotha (model training and API).
