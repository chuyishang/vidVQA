import csv

correct = 0
total = 0
with open(f'./first_run.csv') as f:
    reader = csv.reader(f)
    for row in reader:
        total += 1
        #print(row)
        row = [x.strip() for x in row]
        true_ans = row[-1]
        pred_ans = row[-2]
        if true_ans == pred_ans:
            correct += 1
print(correct/total)