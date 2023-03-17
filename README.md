# MLOps
The Machine Learning (ML) lifecycle consists of many complex components from data ingestion to model deployment, and monitoring. MLOps includes the experimentation, iteration, and continuous improvement of the ML lifecycle.
https://databricks.com/wp-content/uploads/2021/12/MLOps-Cycle.png
# Project Overview:
 ![image](https://user-images.githubusercontent.com/97321212/166896152-011b1e32-12b1-47c7-99e8-848dd4bdd37d.png)
![training_pipeline (1)](https://user-images.githubusercontent.com/97321212/225932800-7152ecb7-6f51-4fbe-8d06-cb6468d9de18.png)


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
![resources](https://user-images.githubusercontent.com/97321212/166206354-bbcaf7cc-6b95-48f1-a1d5-63591455ec4a.JPG)

## Save GitHub Secrets
![env git](https://user-images.githubusercontent.com/97321212/166206291-6fd017a6-3e26-424c-a044-6ed88412ec48.JPG)
![rep secret git](https://user-images.githubusercontent.com/97321212/166206327-26e4f50e-56e1-455e-bf40-4cc8a6010a45.JPG)


# Build and Run Data Bricks Model

## 1. Create and Run a cluster on Azure Data bricks

Runtime = 10.1 ML (includes Apache Spark 3.2.0, Scala 2.12)
![image](https://user-images.githubusercontent.com/97321212/166207220-44541414-2f19-42b3-b716-4c94e4fb632d.png)


## 2. Configure Databricks Repo
![azure_db_repo](https://user-images.githubusercontent.com/97321212/166221473-40855dc1-3ec8-47e0-b3c4-862c541a0450.JPG)



## 3. Run all cells and open MLFLOW model registry
Filename: notebooks/railway_ml_model.py
![model](https://user-images.githubusercontent.com/97321212/166208490-d9d375f1-742c-438a-a98a-4184331b9086.JPG)

# Model Deployment

The following files are created for packaging and deploying the model API service:

1. .github/workflows/main.yaml: the continuous integration and continuous delivery pipeline.

2. manifests/api.yaml: the Kubernetes manifest specifying the desired state of the Kubernetes cluster.

3. service/app: a directory containing a FastAPI web service that consumes the MLFlow model.

4. service/configuration.json: a file specifying the model versions to be used as part of the API service.
5.![image](https://user-images.githubusercontent.com/97321212/166565332-79dc1659-c060-47a3-8243-5084e584f310.png)


5. service/Dockerfile: a Dockerfile used to containerize the service.

6. service/requirements.txt: a file specifying the Python dependencies of the API service.


The workflow to be deployed comprises of three jobs:

## A. Build: 
this job will create a Docker container and register it in ACR. This Docker container will be the API that end-users will consume.

## B. Staging: 
this job will deploy the Docker container to the AKS cluster specified in the GitHub environment called Staging. Once deployed, the model's state will transition to the Staging state in the MLFLow model registry.
## C. Production: 
this job will deploy the Docker container to the AKS cluster specified in the GitHub environment called Production. Once deployed, the model's state will transition to the Production state in the MLFLow model registry.
## TODO

# Model monitoring

For model monitoring Azure Container insights willbe enabled as part of AKS. This will monitor the performance of container workloads deployed to the Kubernetes cluster.

## Model Service

## View model service metrics and logs










