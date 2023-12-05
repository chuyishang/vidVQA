import dataset
import utils
from tqdm import tqdm


def get_siglip_query(question):
    with open('./prompts/base_query.txt') as f:
        prompt = f.read()
        prompt = prompt.replace('insert_question', question)
        output = utils.call_llm(prompt)
    return output
    

def main():
    print("here1")
    data = dataset.get_data("queries.csv")
    ids, choices, queries, answers, vid_names = data
    for i in tqdm(range(len(ids))):
        if i <= 2:
            continue
        video_id = vid_names[i].split(".")[0]
        question = queries[i]
        options = choices[i]
        #video = utils.get_video(f'/shared/shang/datasets/nextqa/videos/{video_id}.mp4')
        query = get_siglip_query(question)
        print(video_id)
        try:
            output = utils.answer_question(video_id, question, options, query)
            with open('./results.csv', 'a') as f:
                f.write(f'{ids[i]},{video_id},{question},"{options}",{output},{answers[i]}\n')
        except:
            with open('./failures.csv', 'a') as f:
                f.write(f'{ids[i]},{video_id},{question},"{options}",{output[0]},{answers[i]}\n')
        
if __name__ == "__main__":
    main()