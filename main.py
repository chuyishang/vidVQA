
import modules
import logging
import answerer
import argparse
import csv
from tqdm import tqdm

from dataset import MyDataset
from vidobj import VideoObj

def main(args):
    logging.basicConfig(filename=args.logging_file, level=logging.INFO)
    dataset = MyDataset(data_path=args.data_path,
                        query_file=args.query_file,
                        start_sample=args.start_sample,
                        max_samples=args.max_samples)

    siglip = modules.SiglipModel(gpu_number=args.gpu1, siglip_model_type="ViT-B-16-SigLIP")
    llava = modules.LLAVA(gpu_number=args.gpu2)
    llm = modules.GPTModel()
    ans = answerer.Answerer(llava, llava, siglip, llm)

    batch_correct = 0
    total_correct = 0
    
    for i in tqdm(range(len(dataset))):
        item = dataset[i]
        video = dataset.construct_video(item)
        try:
            pred = ans.forward(video)
            with open(args.output_file, "a") as outfile:
                outfile.write(f"{item['index']},{item['video_name']},{item['query_type']},{item['query']},{item['answer']},{item['possible_answers']},{pred}\n")
            if pred == item["answer"]:
                print("correct")
                batch_correct += 1
                total_correct += 1
            if i+1 % args.print_interval == 0:
                print("Batch accuracy: ", batch_correct / args.print_interval)
                batch_correct = 0
        except Exception as e:
            print(e)
            pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/shared/shang/datasets/nextqa/videos/")
    parser.add_argument("--query_file", type=str, default="/shared/shang/datasets/nextqa/metadata/test_queries_full.csv")
    parser.add_argument("--max_samples", type=int, default=2000)
    parser.add_argument("--start_sample", type=int, default=0)
    parser.add_argument("--multiprocessing", type=bool, default=True)
    parser.add_argument("--print_interval", type=int, default=10)
    parser.add_argument("--logging", type=bool, default=True)
    parser.add_argument("--logging-file", type=str, default="./log.txt")
    parser.add_argument("--output_file", type=str, default="./output.csv")
    parser.add_argument("--gpu1", type=int, default=0)
    parser.add_argument("--gpu2", type=int, default=1)
    args = parser.parse_args()

    main(args)