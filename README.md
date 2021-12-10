# Flight Delay Prediction
MIDS W261: Machine Learning at Scale | Fall 2021 | Final Project
#### Author: Yao Chen, Toby Petty, Zixi Wang, Ferdous Alam


Introduction:

Flight delays are a widespread problem in the United States and across the globe. The impact of flight delay can be a risk that represents financial losses, the dissatisfaction of passengers, time losses, loss of reputation and bad business relations. The ability to predict flight delays will help to reduce strain on the air travel system and result in significant financial savings for airlines and passengers.

Delayed flights are defined by the Federal Aviation Administration as a flight which arrives or departs more than 15  minutes later than scheduled. In 2019, the arrival delay rate was 19.2% and the departure delay rate was 18.18% in the United States[[1]](https://data.worldbank.org/indicator/IS.AIR.PSGR). The aim of our study is to predict departure delay as a binary yes/no feature, meaning flights that depart 15 minutes or more later than the schedule departure time (referred to as DEP_DEL15 in the flights dataset). As the flight departure time approaches the estimates for delays will increase in accuracy, but be less valuable for airlines and passengers. Given the tradeoff between accuracy and usefulness of delay prediction we have opted to predict the departure delay 2 hours ahead of scheduled departure time.


-  Final Report: [W261_AU21_FINAL_PROJECT.ipynb](https://github.com/yaoc16/Machine-Learning-at-Scale-W261-Final-Project/blob/main/W261_AU21_FINAL_PROJECT.ipynb)
-  [Presentation slides](https://github.com/yaoc16/Machine-Learning-at-Scale-W261-Final-Project/blob/main/Flight%20Delay%20Prediction%20-%20Presentation%20Slides.pdf)



#### Appendix (Code)
1. [EDA Notebook](https://github.com/yaoc16/Machine-Learning-at-Scale-W261-Final-Project/blob/main/EDA%20-%20Full%20Dataset.ipynb)
2. [Full Data Pipeline](https://github.com/yaoc16/Machine-Learning-at-Scale-W261-Final-Project/blob/main/Full%20Data%20Pipeline.ipynb)
3. [CV GridSearch](https://github.com/yaoc16/Machine-Learning-at-Scale-W261-Final-Project/blob/main/CV%20Gridsearch.ipynb)
4. [Model Selection and Ensemble](https://github.com/yaoc16/Machine-Learning-at-Scale-W261-Final-Project/blob/main/Model%20Selection%20and%20Ensemble.ipynb)
5. [Pipeline: Full_Data_Pipeline.py](https://github.com/yaoc16/Machine-Learning-at-Scale-W261-Final-Project/blob/main/Full%20Data%20Pipeline.py)
6. [Pipeline: Model Selection and Ensemble.py](https://github.com/yaoc16/Machine-Learning-at-Scale-W261-Final-Project/blob/main/Model%20Selection%20and%20Ensemble.py)
7. [Pipeline: gridsearch_cv.py](https://github.com/yaoc16/Machine-Learning-at-Scale-W261-Final-Project/blob/main/CV%20Gridsearch.py)
