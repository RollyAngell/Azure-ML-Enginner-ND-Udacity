# Project 2 - Operationalizing Machine Learning

This project is the second project of the Udacity Azure ML Nanodegree. In this project, we deploy a model, consume endpoints to interact with the deployed model in Azure ML Studio and perform pipeline automation to improve machine learning operations.

## Architectural Diagram

![Arquitectura](https://user-images.githubusercontent.com/8076356/107143883-74302f80-6905-11eb-8a6a-ff6955193748.png)

## Key Steps

## STEP 1: Authentication

### AZ Configuration

![image](https://user-images.githubusercontent.com/8076356/107144070-a2623f00-6906-11eb-8bb2-f009e2a86737.png)

*We can see all properties*

![image](https://user-images.githubusercontent.com/8076356/107144094-cc1b6600-6906-11eb-9d06-f5469c5d3034.png)

*AZ login*

![image](https://user-images.githubusercontent.com/8076356/107144105-e1909000-6906-11eb-88f8-171b0c0db179.png)

*az extension add -n azure-cli-ml*

## STEP 2: Automated ML Experiment

### 2.1. Registered Datasets
Here we upload the bank marketing dataset.

![image](https://user-images.githubusercontent.com/8076356/107144668-b0b25a00-690a-11eb-8a53-8f8ddf75a690.png)

*Uploadin dataset*

![image](https://user-images.githubusercontent.com/8076356/107144274-1cdf8e80-6908-11eb-9f59-2d0876e5f8aa.png)

*We can see 2 dataset  and we use firts*

### 2.2. Completed experiment

![image](https://user-images.githubusercontent.com/8076356/107144744-3b935480-690b-11eb-8383-242538bcd9ef.png)

*We run a Automated ML with classification task.*

![image](https://user-images.githubusercontent.com/8076356/107144783-74cbc480-690b-11eb-8550-85307cd7afd5.png)

*Completed experiments*

![image](https://user-images.githubusercontent.com/8076356/107144797-88772b00-690b-11eb-87c3-4255cace6035.png)

*Detail of the experiment*

### 2.3. Best Model

The best model was Voting Ensemble

![image](https://user-images.githubusercontent.com/8076356/107144822-b6f50600-690b-11eb-9905-2d7847d566b4.png)

![image](https://user-images.githubusercontent.com/8076356/107144839-d3913e00-690b-11eb-9d86-8948a446aa7f.png)


## STEP 3: Automated ML Experiment

### 3.1. Deploy the best model
Deploying the model in a Azure Container Instance.

![image](https://user-images.githubusercontent.com/8076356/107149167-24f9f700-6925-11eb-8eb0-2b470b3dfc4a.png)


## STEP 4: Enable logging
We choose the best model for deployment and enable "Authentication" while deploying the model using Azure Container Instance (ACI). The executed code in logs.py enables Application Insights. 

"Application Insights enabled" is disabled before executing logs.py.

![image](https://user-images.githubusercontent.com/8076356/107149214-5ecafd80-6925-11eb-9372-1ffe01447079.png)

"Application Insights enabled" is enabled after executing logs.py.

![image](https://user-images.githubusercontent.com/8076356/107149228-71453700-6925-11eb-9eae-18d58e3787b5.png)

Running log.py

![image](https://user-images.githubusercontent.com/8076356/107149242-85893400-6925-11eb-92f6-a74df3ea74b7.png)

## STEP 5: Swagger Documentation
We using swagger ui to see the input required for an API request in order to obtain predictions from the deployed model. Swagger helps in building, documenting, and consuming RESTful web services. It also explains what types of HTTP requests that an API can consume, like POST and GET.

Default localhost

![image](https://user-images.githubusercontent.com/8076356/107149285-c7b27580-6925-11eb-97f3-a3a5799b38dc.png)

Localhost:8000

![image](https://user-images.githubusercontent.com/8076356/107149300-dac54580-6925-11eb-8b5c-6336e3e46571.png)

GET

![image](https://user-images.githubusercontent.com/8076356/107149329-fc263180-6925-11eb-8459-be99d49e066a.png)

POST

![image](https://user-images.githubusercontent.com/8076356/107149336-0ba57a80-6926-11eb-97c8-0e0c62ed41f5.png)


## STEP 6: Consume model endpoint
Changgin scoring_url and primary key:

![image](https://user-images.githubusercontent.com/8076356/107149396-5cb56e80-6926-11eb-92f1-656f0c325105.png)

The following screenshot demonstrates the output received from the deployed model by running endpoint.py against the REST API

![image](https://user-images.githubusercontent.com/8076356/107149408-6ccd4e00-6926-11eb-9005-4530876b39ff.png)


### STEP 7. Create, publish, and consume a pipeline
Pipeline created
![image](https://user-images.githubusercontent.com/8076356/107867511-8a7c4500-6e49-11eb-9b88-1fdf69ab2f46.png)

Here we show a REST endpoint and a status of ACTIVE.

![image](https://user-images.githubusercontent.com/8076356/107867557-c31c1e80-6e49-11eb-8a3f-c9278de62c28.png)

![image](https://user-images.githubusercontent.com/8076356/107867572-eba41880-6e49-11eb-8f59-449f410f450f.png)

![image](https://user-images.githubusercontent.com/8076356/107867580-0aa2aa80-6e4a-11eb-8550-f57207cc35cf.png)

Completed pipeline run:

![image](https://user-images.githubusercontent.com/8076356/107149477-c897d700-6926-11eb-8731-85c52fb5e461.png)

Here we use RunDetails Widget.

![image](https://user-images.githubusercontent.com/8076356/107867356-53596400-6e48-11eb-8bd5-062ffe453e98.png)

## Screen Recording
Link of youtube

https://youtu.be/d8ScC-0NhUE

## Standout Suggestions

- Adding a step to clean up the data could increase the accuracy.
- We could use an AKS and validate the performance with many requests and see the escalation.
- Adding Deep Learnign capability coud improve the performance.

![image](https://user-images.githubusercontent.com/8076356/107149846-fe3dbf80-6928-11eb-87ad-a127315b373a.png)

