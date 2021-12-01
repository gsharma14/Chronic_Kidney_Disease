# Predicitng Chronic Kidney Disease using patient profile
## Identifying Risk Factors and Subtypes of CKD
**Author:** Gopalika Sharma  

#### Problem Statement
Trying to generate a code repository that can be used across biomedical data science projects. The dataset used for the proof of concept can help physicians better understand chronic kidney disease (CKD) using numerous measurements and biomarkers that have been collected. The dataset can be found at UCI Machine Learning Repository.

#### Task
To write modularised reusable Python code that can be used by other similar datasets and other similar projects to help a physician understand 
1. risk factors for CKD 
2. potential CKD subtypes

#### Observation
1. We have 24 feature attributes with 11 numerical attributes and 13 nominal and one target label.The number of people detected with chronic kidney disease(ckd) is 250 and with no-ckd is 150. In the correlation plot, hemo, pcv and rbcc shows high positive correlation with each other followed by su, bgr which are also in the high end of positive correlation.
2. In order to identify the risk factors for the CKD I  ran a classification model so I can identify the risks through feature importance as it helps us identify which features contributed heavily towards the classification of CKD, hence it identifies the risk factors.The top risk factors are, haemoglobin, packed cell volume, serum cretinine and red blood cells count and sugar as they have high importance value.
3. By running unsupervised k-mean clustering on patient's baseline characteristics among 400 participants I identified 5 novel CKD subgroups that best represent the data pattern.But cluster marked 0,1 and 3 show more probability of being subgroups due to the high count of point distribution.I used the centroids within clusters to find inclined features within them so we know what factors are dominating the various subtypes of CKD.
  3.1 Cluster labelled "0" identifies with high valued features like red blood cells, haemoglobin, packed cell volume, sodium.
  3.2 Cluster labelled "1" identifies with high valued features like serum creatinine, blood urea, albumin, hypertension, diabetes mellitus, sugar, anaemia, blood pressure, pedal adema, blood glucose random, appetite, potassium, coronary artery disease.
  3.3 Cluster labelled "2" identifies with high valued features like potassium, serum creatinine, blood urea, sugar, blood glucose random, sodium, albumin, blood pressure, pedal adema, anaemia, hypertension, diabetes mellitus, age.
  3.4 Cluster labelled "3" identifies with high valued features like blood glucose random, sugar, albumin, hypertension, serum creatinine, diabetes mellitus, blood urea, white blood cells count, appetite.
  3.5 Cluster labelled "4" identifies with high valued features like serum creatinine, blood urea, hypertension, blood glucose random, appetite, coronary artery disease, diabetes and age.


