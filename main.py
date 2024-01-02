from dataset import MyDataset
import modules
from vidobj import VideoObj
import answerer

dataset = MyDataset(data_path="/shared/shang/datasets/nextqa/videos/",
                    query_file="/shared/shang/datasets/nextqa/metadata/splits/val/val_queries_t.csv",
                    max_samples=100)

item = dataset[2]
print(item["query"], item['possible_answers'])
print(item["answer"])
video = VideoObj(item["video"], item["query"], item["possible_answers"])

siglip = modules.SiglipModel(gpu_number=0, siglip_model_type="ViT-B-16-SigLIP")
llava = modules.LLAVA(gpu_number=1)

llm = modules.GPTModel()
ans = answerer.Answerer(llava, llava, siglip, llm, video)
result = ans.forward()
print(result)