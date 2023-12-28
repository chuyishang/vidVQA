from modules import BaseModel
from vidutils import VideoInfo
import abc
from vidutils import VideoObj
from config import settings as config

"""
Answerer design:

Input:
- Video segment or frame
- Question
- Possible Answers
- Global context of video

Answerer Process:
- Frame or video segment is fed into the Answerer, with global context or summary
- Extractor extracts information about current segment
    - Generate questions to ask using LLM
    - Ask questions using VQA
    - Generate captions using captioner
- Evaluator
    - Evaluate if the information is enough to answer the question
    - If yes, answer the question using a Evaluator's final select module
    - If no, pass all current context into Planner
- Planner
    - Read existing information
    - Create new plan for extracting new information
- Retriever
    - Retrieves frames based on plan
    - Sends info back to Extractor
"""
# ========================== Base abstract model ========================== #
class Answerer():
    def __init__(self, caption_model: BaseModel, vqa_model: BaseModel, llm: BaseModel, video_obj: VideoObj):
        self.caption_model = caption_model
        self.vqa_model = vqa_model
        self.llm = llm
        self.video = video_obj

        # TODO: initialize the 4 other classes here

    def construct_prompt(question: str, choices: list, video_info, prompt_path: str) -> str:
        with open(prompt_path) as f:
            prompt = f.read()
        prompt = prompt.replace("INSERT_QUESTION_HERE", question).replace("INSERT_CHOICES_HERE", str(choices)).replace("INSERT_INFO_HERE", str(video_info))
        return prompt




class Extractor(Answerer):
    def __init__(self, caption_model: BaseModel, vqa_model: BaseModel, llm: BaseModel):
        self.caption_model = caption_model
        self.vqa_model = vqa_model
        self.llm = llm
    
    def query_caption(self, frame_number: int, video_info: VideoInfo) -> str:
        frame = video_info[frame_number]
        caption = self.caption_model.caption(image=frame)
        return caption

    def query_VQA(self, question: str, frame_number: int, video_info) -> str:
        frame = video_info[frame_number]
        answer = self.vqa_model.qa(image=frame, question=question)
        return answer

    def forward():
        pass

class Evaluator(Answerer):
    def __init__(self, llm: BaseModel, info: dict, question: str, choices: list):
        self.llm = llm
        self.enough_info = False
    
    @staticmethod
    def construct_prompt(question: str, choices: list, video_info, prompt_path: str) -> str:
        with open(prompt_path) as f:
            prompt = f.read()
        prompt = prompt.replace("INSERT_QUESTION_HERE", question).replace("INSERT_CHOICES_HERE", str(choices)).replace("INSERT_INFO_HERE", str(video_info))
        return prompt
        
    def evaluate_info(self, question: str, choices: list, video_info) -> str:
        prompt_path = config["evaluator"]["enough_info_prompt"]
        prompt =  self.construct_prompt(question, choices, video_info, prompt_path)
        output = self.llm.forward(prompt)
        return output

    def final_select(self, question: str, choices: list, video_info):
        prompt_path = config["evaluator"]["enough_info_prompt"]
        prompt =  self.construct_prompt(question, choices, video_info, prompt_path)
        output = self.llm.forward(prompt)
        return output


class Planner():
    def __init__(self, llm: BaseModel):
        self.llm = llm

    def create_plan():

        

class Retriever():
    def select_frame():
        pass


class Answerer():
    """Main class for answering questions."""
    def __init__(self, caption_model: BaseModel, vqa_model: BaseModel, llm: BaseModel):
        self.caption_model = caption_model
        self.vqa_model = vqa_model
        self.llm = llm

    def construct_prompt(self, question: str, choices: list, video_info: VideoInfo) -> str:
        pass
    
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
        pass

    
        


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