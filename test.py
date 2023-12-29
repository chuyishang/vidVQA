"""import modules
from dataset import MyDataset

dataset = MyDataset(data_path="/shared/shang/datasets/nextqa/videos/",
                    query_file="/shared/shang/datasets/nextqa/metadata/queries_2k.csv",
                    max_samples=5)

blip = modules.BLIPModel(gpu_number=4)
siglip = modules.SiglipModel(gpu_number=3)

item = dataset[3]

keyframe = siglip.forward(images=item['video'], queries=["black bird moving away from pot"])
blip.forward(keyframe[0][0], question="Describe the scene with the most detail possible. Comment on the locations, sizes, and shapes of things as well.", task="qa")
#blip.qa(keyframe[0][0], question="Describe the scene with the most detail possible. Comment on the locations, sizes, and shapes of things as well.")"""
from models.LLaVA.llava.eval import run_llava

model_path = "/home/shang/vidVQA/models/LLaVA/llava-v1.5-7b"
prompt = "What are the things I should be cautious about when I visit here?"
image_file = "https://llava-vl.github.io/static/images/view.jpg"

args = type('Args', (), {
    "model_path": model_path,
    "model_base": None,
    "model_name": run_llava.get_model_name_from_path(model_path),
    "query": prompt,
    "conv_mode": None,
    "image_file": image_file,
    "sep": ",",
    "temperature": 0,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 512
})()

result = run_llava.eval_model(args)
print(result)