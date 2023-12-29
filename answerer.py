import abc
import ast

from modules import BaseModel
from vidobj import VideoObj
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
# ========================== Base answerer model ========================== #
class Answerer():
    def __init__(self, caption_model: BaseModel, vqa_model: BaseModel, similarity_model: BaseModel, llm: BaseModel, video_obj: VideoObj, max_tries=10):
        self.similarity_model = similarity_model
        self.caption_model = caption_model
        self.vqa_model = vqa_model
        self.llm = llm
        self.video = video_obj
        self.max_tries = max_tries

        self.planner = Planner(self)
        self.retriever = Retriever(self)
        self.extractor = Extractor(self)
        self.evaluator = Evaluator(self)
        
    def construct_prompt(question: str, choices: list, video_info, prompt_path: str) -> str:
        with open(prompt_path) as f:
            prompt = f.read()
        prompt = prompt.replace("INSERT_QUESTION_HERE", question).replace("INSERT_CHOICES_HERE", str(choices)).replace("INSERT_INFO_HERE", str(video_info))
        return prompt
    
    def siglip_prompt(self, question: str) -> str: 
        with open(config["answerer"]["siglip_prompt"]) as f:
            prompt = f.read()
        prompt = prompt.replace("INSERT_QUESTION_HERE", question)
        return prompt

    def get_keyframe(self, images, question):
        siglip_prompt = self.siglip_prompt(question)
        keyframe_query = self.llm.forward(siglip_prompt)
        index, keyframe = self.similarity_model.forward(images, queries=[keyframe_query])
        return index.item(), keyframe[0][0]
    
    def construct_info(self, start_sec, end_sec, answer, question=None, type="caption"):
        key = f"Time {start_sec}/{end_sec}"
        if type == "caption":
            return key, answer
        else: #qa case
            qa_pair = f"Q - {question} A - {answer}"
            return key, qa_pair


    def init_update(self):
        start_frame, start_image = self.get_keyframe(self.video.images, self.video.question)
        start_sec, end_sec = self.video.get_second_from_frame(start_frame), self.video.get_second_from_frame(len(self.video))
        caption = self.extractor.forward(start_image, type="caption")

        init_key, init_info = self.construct_info(start_sec, end_sec, caption)
        self.video.info[init_key] = init_info

    def forward(self):
        self.init_update()
        for _ in range(self.max_tries):
            plan = self.planner.forward()
            new_tstmp, frame, questions = self.retriever.forward(plan)
            self.extractor.forward(frame, questions, new_tstmp)
            final_choice = self.evaluator.forward(self.question, self.choices, self.video)
            if final_choice:
                return final_choice
        return self.evaluator.forward(self.question, self.choices, self.video, final_choice=True)



class Extractor(Answerer):
    def __init__(self, answerer):
        self.answerer = answerer

    def get_caption(self, image):
        caption_base = "Describe the image in as much detail as possible."
        caption = self.answerer.vqa_model.forward([image], caption_base)
        return caption
    
    def get_vqa(self, image, question):
        answer = self.answerer.qa_model.forward([image], question)
        return answer

    def forward(self, image, questions=None, start_sec=0, caption=True):
        """Forward function for Extractor. Takes in a single image, and questions as a list"""
        caption_base = "Describe the image in as much detail as possible."
        results = []
        if caption:
            new_caption = self.get_caption(image)
            results.append(caption)
            key, text = self.answerer.construct_info(start_sec, end_sec=self.answerer.video.length_secs, answer=new_caption, type="caption")
            results.append(text)
        if questions:
            for question in questions:
                new_answer = self.get_vqa(image, question)
                key, qa_pair = self.answerer.construct_info(start_sec, end_sec=self.answerer.video.length_secs, answer=new_answer, question=question, type="qa")
                results.append(qa_pair)
        # TODO: check if this is expected behavior
        self.answerer.video.info["key"] = results

class Evaluator(Answerer):
    def __init__(self, answerer):
        self.answerer = answerer
    """@staticmethod
    def construct_prompt(question: str, choices: list, video_info, prompt_path: str) -> str:
        with open(prompt_path) as f:
            prompt = f.read()
        prompt = prompt.replace("INSERT_QUESTION_HERE", question).replace("INSERT_CHOICES_HERE", str(choices)).replace("INSERT_INFO_HERE", str(video_info))
        return prompt
    """
        
    def evaluate_info(self, question: str, choices: list, video) -> str:
        prompt_path = config["evaluator"]["evaluator_prompt"]
        prompt =  self.answerer.construct_prompt(question, choices, video.info, prompt_path)
        output = self.answerer.llm.forward(prompt)
        return output



    def final_select(self, question: str, choices: list, video_info):
        prompt_path = config["evaluator"]["final_select"]
        prompt =  self.answerer.construct_prompt(question, choices, video_info, prompt_path)
        output = self.answerer.llm.forward(prompt)
        return int(output)

    
    def parse_output(self, answer):
        try:
            output = ast.literal_eval(answer)
            final_choice = output[0]
            return final_choice
        except Exception as e:
            print(e)
    
    def forward(self, question: str, choices: list, video, final_choice=False):
        if not final_choice:
            output = self.evaluate_info(question, choices, video)
        else:
            output = self.final_select(question, choices, video.info)
        final_output = self.parse_output(output)
        return final_output

class Planner():
    def __init__(self, answerer):
        self.answerer = answerer

    def create_plan(self, video):
        prompt_path = config["planner"]["planner_prompt"]
        prompt = self.answerer.construct_prompt(video.question, video.choices, video.info, prompt_path)
        output = self.answerer.llm.forward(prompt)
        return output
        
    def clean_output(self, output: list):
        pass
    
    def forward(self, info, question, choices):
        output = self.create_plan(info, question, choices)
        return output

class Retriever():
    def __init__(self, answerer):
        self.answerer = answerer

    # TODO: modify prompt to accept multiple question asking
    def select_frame(self, video, plan):
        prompt_path = config["retriever"]["retriever_prompt"]
        prompt = self.answerer.construct_prompt(video.question, video.choices, video.info, prompt_path).replace("INSERT_PLAN_HERE", plan)
        output = self.answerer.llm.forward(prompt)
        return output
    
    def parse_answer(self, answer):
        try:
            output = ast.literal_eval(answer)
            goto = output['Go-To']
            goto = float(goto)
            questions = output["Questions"]
            frame = self.answerer.video.get_frame_from_second(goto)
            return goto, frame, questions
        except Exception as e:
            print(e)
    
    def forward(self, video, plan):
        output = self.select_frame(video, plan)
        goto, frame, questions = self.parse_answer(output)
        return goto, frame, questions
        





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