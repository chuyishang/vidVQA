from modules import BaseModel
from vidutils import VideoInfo
import ast

class Answerer():
    def __init__(self, caption_model: BaseModel, vqa_model: BaseModel, llm: BaseModel):
        self.caption_model = caption_model
        self.vqa_model = vqa_model
        self.llm = llm

    def construct_prompt(self, question: str, choices: list, video_info: VideoInfo) -> str:
        with open('prompts/base_prompt.txt', 'r') as f:
            prompt = f.read()
        prompt = prompt.replace('insert_question', question)
        prompt = prompt.replace('insert_choices', str(choices))
        
        summary = ''
        for frame_number in sorted(video_info.caption_memory):
            summary += f'- Frame {frame_number}: {video_info.caption_memory[frame_number]}'
            if frame_number in video_info.vqa_memory:
                summary += f', {video_info.vqa_memory[frame_number]}'
            summary += '\n'
        prompt = prompt.replace('insert_summary', summary)
        return prompt
    
    def query_caption(self, frame_number: int, video_info: VideoInfo) -> str:
        frame = video_info[frame_number]
        caption = self.caption_model.caption(image=frame)
        return caption

    def query_VQA(self, question: str, frame_number: int, video_info) -> str:
        frame = video_info[frame_number]
        answer = self.vqa_model.qa(image=frame, question=question)
        return answer

    def select_frame():
        pass

    def get_answer(self, question: str, subquestion: str, choices: list, video_info: VideoInfo, LIMIT: int=10):
        """Main functionality for retrieving an answer to a question."""
        while LIMIT > 0:
            print(f'Iteration: {10 - LIMIT}')
            LIMIT -= 1
            prompt = self.construct_prompt(question, choices, video_info)
            print(prompt)
            response = self.llm.call_llm(prompt)
            print(response)
            parts = response.split('<explanation>')
            explanation = parts[0]
            args = ast.literal_eval(parts[1])
            
            label = args[0]
            number = args[1]
            
            video_info.explanations.append(explanation)
            if label == 'done':
                # number is index of answer choice
                return number
            elif label == 'before':
                frame_numbers = sorted(video_info.caption_memory.keys())
                for i, curr_frame_number in enumerate(frame_numbers):
                    if curr_frame_number == number:
                        prev_frame_number = frame_numbers[i - 1]
                        break
                new_frame_number = int((prev_frame_number + curr_frame_number) / 2)
            elif label == 'after':
                frame_numbers = sorted(video_info.caption_memory.keys())
                for i, curr_frame_number in enumerate(frame_numbers):
                    if curr_frame_number == number:
                        next_frame_number = frame_numbers[i + 1]
                        break
                new_frame_number = int((curr_frame_number + next_frame_number) / 2)
            
            caption = self.query_caption(new_frame_number, video_info)[0]
            video_info.caption_memory[new_frame_number] = caption
            
            # TODO: remove subquestion out of here?
            vqa_answer = self.query_VQA(subquestion, new_frame_number, video_info)[0]
            video_info.vqa_memory[new_frame_number] = vqa_answer
        return
    

    
        


"""
class frame_selector():
    def __init__(self):

    def get_answer(self, images, question, answer_choices, curr_frame, total_frames, caption, prev_info=None):
        LIMIT = 10
        goto_frame = curr_frame
        VQA_question = None
        info = {}
        caption = caption
        while LIMIT >= 0:
            print("LIMIT: ", LIMIT)
            LIMIT -= 1
            if goto_frame == curr_frame:
                # ask a random question.
                question = question
                image = self.vis_processors["eval"](images[curr_frame]).unsqueeze(0).to(modules.device2)
                question = self.txt_processors["eval"](question)
                answer = self.model.predict_answers(samples={"image": image, "text_input": question}, inference_method="generate")
                print("VQA ANSWER: ", answer)
                if answer in answer_choices:
                    return answer
                else:
                    VQA_question = question
                    goto_frame = total_frames - 1
            else:
                # ask a question about the frame
                question = VQA_question
                image = self.vis_processors["eval"](images[goto_frame]).unsqueeze(0).to(modules.device2)
                question = self.txt_processors["eval"](question)
                answer = self.model.predict_answers(samples={"image": image, "text_input": question}, inference_method="generate")
                print("VQA ANSWER: ", answer)
                if answer in answer_choices:
                    return answer
                else:
                    goto_frame -= 1
            if goto_frame == curr_frame:
                # ask a random question.
                image = self.vis_processors2["eval"](images[curr_frame]).unsqueeze(0).to(modules.device2)
                caption = self.model2.generate({"image": image})
                caption = caption[0]
                print("CAPTION: ", caption)
                if caption in answer_choices:
                    return caption
                else:
                    goto_frame = total_frames - 1
            else:
                # ask a question about the frame
"""