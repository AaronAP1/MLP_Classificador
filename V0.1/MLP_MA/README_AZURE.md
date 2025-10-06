# MLP_MA on Azure ML

This folder contains a minimal Azure ML job spec to train the classifier in the cloud.

## Files
- `azure/environment.yml`: Conda env, installs Python 3.12 and pip deps from `MLP_MA/requirements.txt`.
- `azure/job.yml`: Azure ML command job definition.

## Submit a job (CLI v2)

1. Log in and set workspace (once):

```powershell
az login
az account set --subscription "<SUBSCRIPTION>"
az configure --defaults workspace="<WS_NAME>" group="<RG_NAME>"
```

2. Create a compute (once):

```powershell
az ml compute create --name cpu-cluster --size Standard_D4s_v5 --type AmlCompute --min-instances 0 --max-instances 1
```

3. Upload config/paths as data assets:

```powershell
az ml data create --name config-adultos --path MLP_MA/configs/config_adultos.json --type uri_file
az ml data create --name paths-adultos  --path MLP_MA/configs/paths_adultos.json  --type uri_file
# Repeat for jovenes, ninos as needed
```

4. Submit training (example: Adultos):

```powershell
az ml job create --file MLP_MA/azure/job.yml \
  --set inputs.config=azureml:config-adultos \
        inputs.paths=azureml:paths-adultos \
  --name mlp-ma-adultos-$(Get-Date -Format yyyyMMddHHmmss)
```

Outputs are written under `MLP_MA/outputs` inside the run’s snapshot; use `az ml job download --name <job_name>` to fetch artifacts.

## Notes
- AMP runs only when CUDA is available; the job uses CPU image by default. Switch to a GPU compute and image for faster training.
- Ensure raw CSVs are accessible within the repo path (e.g., `MLP_Adul_s/data/raw/adultos.csv`).
