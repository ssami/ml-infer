BentoML:
 
* Treats the model as first-class citizen; code is expected to be bundled with the model. 
But this means the APIs are associated to the trained model. 
Sidesteps the question of code & model getting out of sync. 
* Lots of annotations that can auto-define model type, API and so on 
* Has a model registry (but not experimentation management) 
* CLI commands 
* It generates a Dockerfile that can be used to serve the API, woo (slightly involved, https://docs.bentoml.org/en/latest/concepts.html#api-server-dockerization)
* No client but different ways to access the service itself (https://docs.bentoml.org/en/latest/concepts.html#model-serving) 
* Nice thing is that the YataiService UI has its own approved Docker image with each release… wish MLFlow had the same 
* Can deploy to a bunch of different platforms, including Clipper and AWS
* Seems like you can NOT load just the model from S3; need to change the Yatai endpoint to point to S3. 
Bit convoluted: https://docs.bentoml.org/en/latest/concepts.html?highlight=s3#customizing-model-repository. 

Results: 
```
INFO - Successfully saved BentoService bundle 'DiabetesRegressor:20200818161920_8B974A' to S3: s3://<bucket-name>/DiabetesRegressor/20200818161920_8B974A.tar.gz
```
The various versions of the service are saved in the bucket named after the service. 

Saving and making Dockerfile: 

```
bentoml get DiabetesRegressor:latest                                                                                                                                         4 ↵
[2020-08-18 16:23:49,826] INFO - Getting latest version DiabetesRegressor:20200818162141_93CCA4
[2020-08-18 16:23:49,826] ERROR - RPC ERROR GetBento: Bento DiabetesRegressor:20200818162141_93CCA4 not found in target repository
Error: bentoml-cli get failed: INTERNAL:Bento DiabetesRegressor:20200818162141_93CCA4 not found in target repository
```

I don't think we can get the URI from a remote bundle, so that'll have to be saved locally as well. 


Trying to save after saving to s3:// will result in: 
```
[2020-08-18 16:25:36,334] ERROR - RPC ERROR GetBento: Bento DiabetesRegressor:20200818162535_09D91B not found in target repository
Traceback (most recent call last):
  File "/Users/ssami/anaconda3/envs/ml-infer/lib/python3.8/runpy.py", line 194, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "/Users/ssami/anaconda3/envs/ml-infer/lib/python3.8/runpy.py", line 87, in _run_code
    exec(code, run_globals)
  File "/Users/ssami/git/ssami/ml-infer/bento/service.py", line 33, in <module>
    service.save()
  File "/Users/ssami/anaconda3/envs/ml-infer/lib/python3.8/site-packages/bentoml/service.py", line 782, in save
    return save(self, base_path, version)
  File "/Users/ssami/anaconda3/envs/ml-infer/lib/python3.8/site-packages/bentoml/service.py", line 470, in save
    return yatai_client.repository.upload(bento_service, version)
  File "/Users/ssami/anaconda3/envs/ml-infer/lib/python3.8/site-packages/bentoml/yatai/client/bento_repository_api.py", line 60, in upload
    return self._upload_bento_service(bento_service, tmpdir)
  File "/Users/ssami/anaconda3/envs/ml-infer/lib/python3.8/site-packages/bentoml/yatai/client/bento_repository_api.py", line 80, in _upload_bento_service
    raise BentoMLException(
bentoml.exceptions.BentoMLException: Failed accessing YataiService. INTERNAL:Bento DiabetesRegressor:20200818162535_09D91B not found in target repository

```
So I'd have to save that separately to a local env, then somehow override the version: 
```
Traceback (most recent call last):
  File "/Users/ssami/anaconda3/envs/ml-infer/lib/python3.8/runpy.py", line 194, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "/Users/ssami/anaconda3/envs/ml-infer/lib/python3.8/runpy.py", line 87, in _run_code
    exec(code, run_globals)
  File "/Users/ssami/git/ssami/ml-infer/bento/service.py", line 34, in <module>
    service.save(base_path='s3://ssami-1577132529')
  File "/Users/ssami/anaconda3/envs/ml-infer/lib/python3.8/site-packages/bentoml/service.py", line 782, in save
    return save(self, base_path, version)
  File "/Users/ssami/anaconda3/envs/ml-infer/lib/python3.8/site-packages/bentoml/service.py", line 470, in save
    return yatai_client.repository.upload(bento_service, version)
  File "/Users/ssami/anaconda3/envs/ml-infer/lib/python3.8/site-packages/bentoml/yatai/client/bento_repository_api.py", line 60, in upload
    return self._upload_bento_service(bento_service, tmpdir)
  File "/Users/ssami/anaconda3/envs/ml-infer/lib/python3.8/site-packages/bentoml/yatai/client/bento_repository_api.py", line 72, in _upload_bento_service
    raise BentoMLException(
bentoml.exceptions.BentoMLException: BentoService bundle DiabetesRegressor:20200818162705_55244E already registered in repository. Reset BentoService version with BentoService#set_version or bypass BentoML's model registry feature with BentoService#save_to_dir

```
To run the Docker container: 
```
python service.py
saved_path=$(bentoml get DiabetesRegressor:latest -q | jq -r ".uri.uri")
docker build -t 410318598490.dkr.ecr.us-east-1.amazonaws.com/diabetes_regressor_bento_service $saved_path
docker run -p 5000:5000 410318598490.dkr.ecr.us-east-1.amazonaws.com/diabetes_regressor_bento_service:latest
```
Saved path looks like `<relative path> / DiabetesRegressor/20200820161343_2F0AA9`

Why doesn't BentoML use Numpy inputs? Why Pandas Dataframe only? DFs are significantly heavier. Issue is open against BentoML
right now to do exactly that. 
