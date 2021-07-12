# Capstone - Azure Machine Learning Engineer

In this project I will create two models: one using Automated ML and one customized model whose hyperparameters are tuned using HyperDrive. I will then compare the performance of both the models and deploy the best performing model.

## Dataset

### Dataset: Heart failure clinical records

I used the 'Heart Failure Clinical Data' because cardiovascular diseases (CVDs) are the number 1 cause of death globally, taking an estimated 17.9 million lives each year, which accounts for 31% of all deaths worlwide.

### Overview
Abstract: This dataset contains the medical records of 299 patients who had heart failure, collected during their follow-up period, where each patient profile has 13 clinical features.

![image](https://user-images.githubusercontent.com/8076356/125221252-96035900-e28d-11eb-8989-f4b27631f71a.png)

Source:

Provide the names, email addresses, institutions, and other contact information of the donors and creators of the data set.The original dataset version was collected by Tanvir Ahmad, Assia Munir, Sajjad Haider Bhatti, Muhammad Aftab, and Muhammad Ali Raza (Government College University, Faisalabad, Pakistan) and made available by them on FigShare under the Attribution 4.0 International (CC BY 4.0: freedom to share and adapt the material) copyright in July 2017.

The current version of the dataset was elaborated by Davide Chicco (Krembil Research Institute, Toronto, Canada) and donated to the University of California Irvine Machine Learning Repository under the same Attribution 4.0 International (CC BY 4.0) copyright in January 2020. Davide Chicco can be reached at <davidechicco '@' davidechicco.it>

### Task

Attribute Information:

Thirteen (13) clinical features:

- age: age of the patient (years)
- anaemia: decrease of red blood cells or hemoglobin (boolean)
- high blood pressure: if the patient has hypertension (boolean)
- creatinine phosphokinase (CPK): level of the CPK enzyme in the blood (mcg/L)
- diabetes: if the patient has diabetes (boolean)
- ejection fraction: percentage of blood leaving the heart at each contraction (percentage)
- platelets: platelets in the blood (kiloplatelets/mL)
- sex: woman or man (binary)
- serum creatinine: level of serum creatinine in the blood (mg/dL)
- serum sodium: level of serum sodium in the blood (mEq/L)
- smoking: if the patient smokes or not (boolean)
- time: follow-up period (days)
- [target] death event: if the patient deceased during the follow-up period (boolean)

### Access

First I upload 'Heart Failure Clinical Data' dataset to my github.

![image](https://user-images.githubusercontent.com/8076356/125230535-8fc9a880-e29e-11eb-9dcb-df9fb0a1265b.png)

Then I validate if data exist in datasets in Azure Machine Learning if not I upload it using TabularDatasetFactory.from_delimited_files.

dataset = Dataset.Tabular.from_delimited_files("https://raw.githubusercontent.com/RollyAngell/ML-Azure-Udacity/main/Project%203%20-%20Casptone%20Project%20AutoML%20vs%20HyperDrive/heart%20failure%20clinical%20records%20dataset.csv")

![image](https://user-images.githubusercontent.com/8076356/125230162-e71b4900-e29d-11eb-8ea7-35c85328060a.png)

![image](https://user-images.githubusercontent.com/8076356/125230217-01edbd80-e29e-11eb-879a-c3a6c532c133.png)

![image](https://user-images.githubusercontent.com/8076356/125230251-1336ca00-e29e-11eb-961b-1fe2b53acca5.png)

## Automated ML

Setting
- Experiment_timeout_minutes:	I choose 30 minutos that means the experiment will exit after that time.
- Max_concurrent_iterations:	To get result quickly we choose 5 that means that 5 iterations at the same time.
- Primary_metric:	This is the metric that we want to optimize (accuracy). I will use 'AUC_weighted' as 'primary_metric' parameter. AUC means the area under the Receiver Operating Characteristic Curve which plots the relationship between true positive rate and false positive rate. Since our dataset doesn't have high class imbalance, we can use ROC method for judging the performance of a model. I will use AUC_weighted in order to mitigate the effects of whatever little imbalance is there in the dataset. AUC_weighted is the arithmetic mean of the score for each class, weighted by the number of true instances in each class.
- Task:	classification
- Compute_target:	To define the compute cluster we will be using
- training_data:	It is the training dataset to be used for the experiment.
- label_column_name:	To specify the dependent variable that we are trying to classify. I will enter 'DEATH_EVENT' as 'label_column_name' parameter.
- path:	To specify the dependent variable that we are trying to classify
- enable_early_stopping:	we can choose to terminate the experiment if the score stops improving in the short term. I will enter 'True' as 'enable_early_stopping' parameter.
- featurization:	To specify the dependent variable that we are trying to classify
- debug_log:	To specify the dependent variable that we are trying to classify
- enable_onnx_compatible_models:	To specify the dependent variable that we are trying to classify


![image](https://user-images.githubusercontent.com/8076356/125231110-c522c600-e29f-11eb-98e9-d398db4a68b7.png)

![image](https://user-images.githubusercontent.com/8076356/125231502-7cb7d800-e2a0-11eb-957c-4bce8afcd495.png)

### Results
AutoML checked for the 'Class Balancing Detection', 'Missing feature values imputation' and 'High Cardinality Feature Detection' for the dataset. 

- Class Balancing Detection  --> Your inputs were analyzed, and all classes are balanced in your training data.
- Missing feature values imputation  --> No feature missing values were detected in the training data.
- High Cardinality Feature Detection  --> Your inputs were analyzed, and no high cardinality features were detected.

AutoML performed 83 iterations out of which it produced VotingEnsemble as the best model with a metric score of 0.9210.

VotingEnsemble is an ensemble model which combines multiple models to improve machine learning results. 
It does so by predicting output on the weighted average of predicted class probabilities. 

estimators: ['64', '58', '66', '44', '76', '54', '29', '72'],
weights: [0.1111111111111111, 0.1111111111111111, 0.1111111111111111, 0.1111111111111111, 0.1111111111111111, 0.2222222222222222, 0.1111111111111111, 0.1111111111111111]

![image](https://user-images.githubusercontent.com/8076356/125241880-a4fc0280-e2b1-11eb-86b7-8d4d78087493.png)

![image](https://user-images.githubusercontent.com/8076356/125241968-c4932b00-e2b1-11eb-899e-a5ffa36dd88a.png)

![image](https://user-images.githubusercontent.com/8076356/125242044-db398200-e2b1-11eb-831d-3c43cf1ba406.png)

![image](https://user-images.githubusercontent.com/8076356/125242118-ea203480-e2b1-11eb-88d1-7257b0cf56cd.png)

![image](https://user-images.githubusercontent.com/8076356/125242184-fa381400-e2b1-11eb-84ef-edd2dc0fbed6.png)
![image](https://user-images.githubusercontent.com/8076356/125242237-091ec680-e2b2-11eb-860a-3dad75636d7b.png)

![image](https://user-images.githubusercontent.com/8076356/125242459-56029d00-e2b2-11eb-92c3-de2a680275f5.png)

![image](https://user-images.githubusercontent.com/8076356/125242510-6a469a00-e2b2-11eb-9d05-27fccf98fbc9.png)

Getting best model

![image](https://user-images.githubusercontent.com/8076356/125242582-821e1e00-e2b2-11eb-9ca7-20a947d836ab.png)

![image](https://user-images.githubusercontent.com/8076356/125242614-8f3b0d00-e2b2-11eb-9d4a-da6159e8ad37.png)
![image](https://user-images.githubusercontent.com/8076356/125242663-9eba5600-e2b2-11eb-94e3-4f2555621732.png)
![image](https://user-images.githubusercontent.com/8076356/125242708-aa0d8180-e2b2-11eb-8bde-604c82eb46a0.png)

## Hyperparameter Tuning

Setting
- estimator:	I have defined an SKLearn estimator below as est and I will use it as estimator parameter.
- hyperparameter_sampling:	It is the sampler that will create the instance of hyperparameters to be used for each sample run. I have defined a RandomParameterSampling below as 'param_sampling' and I will use it as 'hyperparameter_sampling' parameter.
- policy:	It is the early termination policy that will be used to terminate the experiment if no improvement in primary metric is witnessed after some runs. I have defined a BanditPolicy below as 'et_policy' and I will use it as 'policy' parameter.
- primary_metric_name:	it is the name of the metric on the basis of which performance of different models will be judged.
- primary_metric_goal:	In order to get the best model for our classification task, my goal is to maximize the AUC_weighted metric hence I will enter 'PrimaryMetricGoal.MAXIMIZE'as 'primary_metric_goal' parameter.
- max_total_runs:	It is the maximum number of child runs that will be executed in the experiment to find the best model for the task intended. I will enter '25' as the 'max_total_runs' parameter which will produce a good and acceptable result in less amount of time.

![image](https://user-images.githubusercontent.com/8076356/125235354-f1424500-e2a7-11eb-9e78-66846594eb4d.png)

### Results
The best performing Hyperdrive model was a Logistic Regression model with parameter value of 
C = '0.5633339376963704' 
max_iter = '100'
Accuracy = 0.7575757575757576.

![image](https://user-images.githubusercontent.com/8076356/125239739-b7287180-e2ae-11eb-92f9-300c0a3121b8.png)

![image](https://user-images.githubusercontent.com/8076356/125239995-16868180-e2af-11eb-8e33-796374756f38.png)

![image](https://user-images.githubusercontent.com/8076356/125240198-69f8cf80-e2af-11eb-8da8-16c0be741c9f.png)

![image](https://user-images.githubusercontent.com/8076356/125240289-88f76180-e2af-11eb-88d4-89542ae11b02.png)

![image](https://user-images.githubusercontent.com/8076356/125240332-9b719b00-e2af-11eb-9da1-042b530e2939.png)

## Model Deployment
Deploying in ACI

![image](https://user-images.githubusercontent.com/8076356/125242921-f5c02b00-e2b2-11eb-8a36-687b6248d841.png)
![image](https://user-images.githubusercontent.com/8076356/125242976-0b355500-e2b3-11eb-891a-eb2e4fccc98c.png)

In azure machine learning workspace.
![image](https://user-images.githubusercontent.com/8076356/125243026-1f795200-e2b3-11eb-97f6-2eea18400483.png)

I send a request to the web service you deployed to test it
![image](https://user-images.githubusercontent.com/8076356/125244344-d6c29880-e2b4-11eb-8f59-7efc6f577d71.png)


I print the logs of the web service and delete the service
![image](https://user-images.githubusercontent.com/8076356/125244403-e80ba500-e2b4-11eb-887c-f18d34ce9f02.png)

Azure ML show how to consume the web service

![image](https://user-images.githubusercontent.com/8076356/125244498-05407380-e2b5-11eb-8b7a-78dc3e8b9ab9.png)

I proved that code and works well
![image](https://user-images.githubusercontent.com/8076356/125244568-19847080-e2b5-11eb-964c-dd3c59e40c41.png)


## Screen Recording
Link de Youtube

## Standout Suggestions
AutoML works a lot in feature engineering process and that help ML algorithms learn better.
For the HyperDrive run, different algorithms such as Random Forest or Voting Ensemble can be used to obtain a better accuracy.
For AutoML give more time for experiment_timeout_minutes can increase accuracy.

