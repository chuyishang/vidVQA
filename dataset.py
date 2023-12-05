import csv

def get_data(filename):
    ids = []
    choices =  []
    queries = []
    answers = []
    vid_names = []
    with open(f'/shared/shang/datasets/nextqa/metadata/{filename}') as f:
        spamreader = csv.reader(f)
        for row in spamreader:
            #print(row)
            row = [x.strip() for x in row]
            sample_id,possible_answers,query_type,query,answer,video_name = row[1], row[2], row[3], row[4], row[5], row[6]
            ids.append(sample_id)
            choices.append(possible_answers)
            queries.append(query)
            answers.append(answer)
            vid_names.append(video_name)
    return ids, choices, queries, answers, vid_names
    
