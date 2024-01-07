
import modules
import logging
import answerer
import argparse
import csv
from tqdm import tqdm
import torch.multiprocessing as mp

from queue import Empty
from dataset import MyDataset
from vidobj import VideoObj


def get_answer(ans, idx, video):
        correct = False
        pred = ans.forward(video)
        if pred == video.answer:
            print("correct")
            correct = True
        return idx, pred, correct, video
        
def worker(input_queue, output_queue, gpu1, gpu2):
    print("starting process")
    siglip = modules.SiglipModel(gpu_number=gpu1, siglip_model_type="ViT-B-16-SigLIP")
    llava = modules.LLAVA(gpu_number=gpu2)
    llm = modules.GPTModel()
    ans = answerer.Answerer(llava, llava, siglip, llm)
    total_correct = 0
    total_count = 0
    while True:
        if i+1 % args.print_interval == 0:
            print("Total Accuracy:", total_correct / total_count)
        try:
            # idx, video = input_queue.get(timeout=1)
            idx, item = input_queue.get(timeout=1)
            video = dataset.construct_video(item)
        except Empty:
            break
        try:
            result = get_answer(ans, idx, video)
            total_count += 1
            if result[2]:
                total_correct += 1
        except Exception as e:
            print(e)
            input_queue.put((idx, video))
        output_queue.put(result)
    
def printer(output_queue, result_file):
    print("starting printer")
    with open(result_file, "a") as outfile:
        while True:
            try:
                idx, pred, correct, video = output_queue.get(timeout=1)
            except Empty:
                break
            with open(args.output_file, "a") as outfile:
                outfile.write(f"{idx},{video.vid_id},{video.query_type},{video.question},{video.answer},{str(video.choices)},{pred}\n")
        
def main(args):
    mp.set_start_method('spawn')
    logging.basicConfig(filename=args.logging_file, level=logging.INFO)
    dataset = MyDataset(data_path=args.data_path,
                        query_file=args.query_file,
                        start_sample=args.start_sample,
                        max_samples=args.max_samples)

    input_queue = mp.Queue()
    output_queue = mp.Queue()
    print("loading queue")
    #for i in tqdm(range(args.start_sample, max(args.max_samples, len(dataset)))):
    #   item = dataset[i]
    #   video = dataset.construct_video(item)
    #   input_queue.put((i, video))
    for i in tqdm(range(args.start_sample, max(args.max_samples, len(dataset)))):
        item = dataset[i]
        input_queue.put((i, item))
    
    
    
    result_process = mp.Process(target=printer, args=(output_queue, args.output_file))
    result_process.start()
    
    processes = []
    for i in range(args.num_workers):
        # 0 -> {0, 1}, 1 -> {2, 3}, 2 -> {4, 5}, 3 -> {6, 7}, etc.
        p = mp.Process(target=worker, args=(input_queue, output_queue, 2*i, 2*i+1))
        processes.append(p)
        p.start()
    
    for p in processes:
        p.join()

    output_queue.put('DONE')
    result_process.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/shared/shang/datasets/nextqa/videos/")
    parser.add_argument("--query_file", type=str, default="/shared/shang/datasets/nextqa/metadata/test_queries_full.csv")
    parser.add_argument("--max_samples", type=int, default=50)
    parser.add_argument("--start_sample", type=int, default=0)
    parser.add_argument("--multiprocessing", type=bool, default=True)
    parser.add_argument("--print_interval", type=int, default=10)
    parser.add_argument("--logging", type=bool, default=True)
    parser.add_argument("--logging_file", type=str, default="./log.txt")
    parser.add_argument("--output_file", type=str, default="./output.csv")
    parser.add_argument("--gpu1", type=int, default=0)
    parser.add_argument("--gpu2", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=3)
    args = parser.parse_args()

    main(args)