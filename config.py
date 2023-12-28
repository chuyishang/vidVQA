#!/usr/bin/env python
import yaml

with open("/home/shang/vidVQA/configs/base_config.yaml") as f:
    settings = yaml.safe_load(f)


print(settings.keys())
print(settings["gpt"]["qa_prompt"])
