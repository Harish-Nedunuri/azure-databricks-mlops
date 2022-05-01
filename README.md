# MLOps
The Machine Learning (ML) lifecycle consists of many complex components from data ingest to model deployment, and monitoring. MLOps includes the experimentation, iteration, and continuous improvement of the ML lifecycle.
https://databricks.com/wp-content/uploads/2021/12/MLOps-Cycle.png
# Project Overview:
 ![image](https://user-images.githubusercontent.com/97321212/166095207-d89e2b18-e9d9-450c-80fd-652b03d95d69.png)
## Tech stack used 
1. Azure databricks- For Model development  and management
2. MLFlow- ML experiment tracking, registration, & deployment
3. Azure Kubernetes Service - to deploy containers as web services 
4. Azure Container Registry to manage and store docker containers
5. Azure Log Analytics- Azure monitor
6. GitHub- automation Repo, secrets for CI & CD
7. FastAPI- For model webservice
8. Programming 

    a. Azure bicep- Azure ARM deployment 
       
    b. Python- databricks notebook with model  and fastAPI webservice functions
    
    c. yaml scripts - for GIT actions
 

## Project Objectives
1. Develop resources for an prototype ML project which illustrates the following
    a. How an MLFlow model can be trained on Databricks, 
    b. packaged as a web service, 
    c. deployed to Kubernetes via CI/CD, and 
    d. monitored within Microsoft Azure
2. Develop and register a ML model to predict likelihood of railway track degradation.

# Deploy Azure Resources

#To deploy the resources use the link below

https://portal.azure.com/#create/Microsoft.Template/uri/https%3A%2F%2Fraw.githubusercontent.com%2FHarish-Nedunuri%2Fazure-databricks-mlops%2Fmain%2Fazure_infrastructure%2Fmain.json
![deployment failed](https://user-images.githubusercontent.com/97321212/166141017-2024d8fc-df05-4726-910b-ecbd2dd74352.JPG)

