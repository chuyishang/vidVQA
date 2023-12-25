import modules
from dataset import MyDataset

dataset = MyDataset(data_path="/shared/shang/datasets/nextqa/videos/",
                    query_file="/shared/shang/datasets/nextqa/metadata/queries_2k.csv",
                    max_samples=5)

blip = modules.BLIPModel(gpu_number=4)
siglip = modules.SiglipModel(gpu_number=3)

item = dataset.get[1]

keyframe = siglip.forward(images=item['video'], queries=["black bird moving away from pot"])
#blip.forward(keyframe[0][0], question="Describe the scene with the most detail possible. Comment on the locations, sizes, and shapes of things as well.", task="qa")
#blip.qa(keyframe[0][0], question="Describe the scene with the most detail possible. Comment on the locations, sizes, and shapes of things as well.")