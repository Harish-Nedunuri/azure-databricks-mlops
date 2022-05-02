from pprint import pprint

client=MlflowClient()
[pprint(mv) for mv in client.search_model_versions("name=Railway-attrition")]