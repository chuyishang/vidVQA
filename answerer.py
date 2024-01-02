import abc
import ast
from modules import BaseModel
from vidobj import VideoObj
from config import settings as config
import logging
import PIL

logging.basicConfig(filename='debugging.log', encoding='utf-8', level=logging.INFO)
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
    """Main Answerer class. Contains all the modules and functions to run the Answerer."""
    def __init__(self, caption_model: BaseModel, vqa_model: BaseModel, similarity_model: BaseModel, llm: BaseModel, video_obj: VideoObj, max_tries: int = 10):
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
        
    def construct_prompt(self, question: str, choices: list, video_info, prompt_path: str) -> str:
        with open(prompt_path) as f:
            prompt = f.read()
        prompt = prompt.replace("INSERT_QUESTION_HERE", question).replace("INSERT_CHOICES_HERE", str(choices)).replace("INSERT_INFO_HERE", str(video_info))
        return prompt
    
    def siglip_prompt(self, question: str) -> str: 
        with open(config["answerer"]["siglip_prompt"]) as f:
            prompt = f.read()
        prompt = prompt.replace("INSERT_QUESTION_HERE", question)
        return prompt

    def get_keyframe(self, images: list[PIL.Image.Image], question: str) -> (int, PIL.Image.Image):
        """Uses SigLIP to get the keyframe and index of the keyframe"""
        siglip_prompt = self.siglip_prompt(question)
        keyframe_query = self.llm.forward(siglip_prompt)
        index, keyframe = self.similarity_model.forward(images, queries=[keyframe_query])
        return index.item(), keyframe[0][0]
    
    def construct_info(self, start_sec: float, end_sec: float, answer: str, question: str = None, type: str = "caption") -> (str, str):
        key = f"Time {start_sec}/{end_sec}"
        if type == "caption":
            return key, answer
        else: #qa case
            qa_pair = f"Q - {question} A - {answer}"
            return key, qa_pair


    def init_update(self) -> None:
        """Function that initializes the video info with the first keyframe and caption"""
        start_frame, start_image = self.get_keyframe(self.video.images, self.video.question)
        start_sec, end_sec = self.video.get_second_from_frame(start_frame), self.video.get_second_from_frame(len(self.video))
        key, caption = self.extractor.forward(start_image, start_sec=start_sec)
        #init_key, init_info = self.construct_info(start_sec, end_sec, answer=caption)
        self.video.info[key] = caption

    # TODO: Check if this is correct (do we want int or str?)
    def forward(self) -> int:
        self.init_update()
        for i in range(self.max_tries):
            logging.info(f"TRY: {i}/{self.max_tries}")
            # Planner
            plan, explanation = self.planner.forward(self.video)
            logging.info(f'PLAN: {plan}\nEXPLANATION: {explanation}')
            # Retriever
            new_tstmp, frame, questions, explanation = self.retriever.forward(self.video, plan)
            logging.info(f'NEW TIME: {new_tstmp}, NEW FRAME: {frame}, NEW QUESTIONS: {questions}, EXPLANATION: {explanation}')
            # Extractor
            key, ans = self.extractor.forward(frame[1], questions, new_tstmp)
            # In case we choose the same frame
            if key in self.video.info:
                self.video.info[key].append(ans)[1:]
            else:
                self.video.info[key] = ans
            logging.info(f"INFO: {self.video.info}")
            # Evaluator
            final_choice = self.evaluator.forward(self.video.question, self.video.choices, self.video)
            logging.info(f"FINAL CHOICE: {final_choice}")
            if final_choice:
                return final_choice
        return self.evaluator.forward(self.video.question, self.video.choices, self.video, final_choice=True)

class Extractor(Answerer):
    def __init__(self, answerer):
        self.answerer = answerer

    def get_caption(self, image: PIL.Image.Image) -> str:
        caption_base = "Describe the image in as much detail as possible."
        caption = self.answerer.vqa_model.forward([image], caption_base)
        return caption
    
    def get_vqa(self, image: PIL.Image.Image, question: str) -> str:
        answer = self.answerer.vqa_model.forward([image], question)
        return answer

    def forward(self, image: PIL.Image.Image, questions: list[str] = None, start_sec: int = 0, caption: bool = True) -> (str, str):
        """Forward function for Extractor. Takes in a single image, and questions as a list. By default, captions the image."""
        results = []
        if caption:
            new_caption = self.get_caption(image)
            #results.append(new_caption)
            key, text = self.answerer.construct_info(start_sec, end_sec=self.answerer.video.length_secs, answer=new_caption, type="caption")
            results.append(text)
        if questions:
            for question in questions:
                new_answer = self.get_vqa(image, question)
                key, qa_pair = self.answerer.construct_info(start_sec, end_sec=self.answerer.video.length_secs, answer=new_answer, question=question, type="qa")
                results.append(qa_pair)
        # TODO: check if this is expected behavior
        return key, results
        #self.answerer.video.info[key] = results

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
        
    def evaluate_info(self, question: str, choices: list, video: VideoObj) -> (int, str):
        prompt_path = config["evaluator"]["evaluator_prompt"]
        prompt =  self.answerer.construct_prompt(question, choices, video.info, prompt_path)
        output = self.answerer.llm.forward(prompt)
        return output

    def final_select(self, question: str, choices: list, video_info: dict):
        prompt_path = config["evaluator"]["final_select"]
        prompt =  self.answerer.construct_prompt(question, choices, video_info, prompt_path)
        output = self.answerer.llm.forward(prompt)
        return output

    def parse_output(self, answer: tuple[int, str]) -> int:
        try:
            output = ast.literal_eval(answer)
            # final_choice = output[0]
            final_choice = output["Answer"]
            logging.info(f"Explanation: {output['Explanation']}")
            if final_choice not in [None, "None"]:
                final_choice = int(final_choice)
            return final_choice
        except Exception as e:
            print(e)
    
    def forward(self, question: str, choices: list, video: VideoObj, final_choice: bool = False):
        if not final_choice:
            output = self.evaluate_info(question, choices, video)
        else:
            output = self.final_select(question, choices, video.info)
        final_output = self.parse_output(output)
        return final_output

class Planner():
    def __init__(self, answerer):
        self.answerer = answerer
        self.plan = None

    def create_plan(self, video: VideoObj):
        prompt_path = config["planner"]["planner_prompt"]
        prompt = self.answerer.construct_prompt(video.question, video.choices, video.info, prompt_path).replace("INSERT_PAST_PLAN_HERE", str(self.plan))
        output = self.answerer.llm.forward(prompt)
        output = ast.literal_eval(output)
        self.plan = output["Plan"]
        # modification for explanation
        # return output[0], output[1]
        return output["Plan"], output["Explanation"]
        
    def clean_output(self, output: list[str]) -> list[str]:
        pass
    
    def forward(self, video: VideoObj) -> list[str]:
        output, explanation = self.create_plan(video)
        # explanation modification
        if type(output) != str:
            output = str(output)
        return output, explanation

class Retriever():
    def __init__(self, answerer):
        self.answerer = answerer
        self.curr = None

    # TODO: modify prompt to accept multiple question asking
    def select_frame(self, video: VideoObj, plan: list[str]) -> dict[str, str]:
        prompt_path = config["retriever"]["retriever_prompt"]
        prompt = self.answerer.construct_prompt(video.question, video.choices, video.info, prompt_path).replace("INSERT_PLAN_HERE", plan).replace("INSERT_CURR_HERE", str(self.curr))
        output = self.answerer.llm.forward(prompt)
        return output
    
    def parse_answer(self, answer: dict[str, str]) -> (float, PIL.Image.Image, list[str]):
        try:
            output = ast.literal_eval(answer)
            goto = output['Go-To']
            goto = float(goto)
            questions = output["Questions"]
            explanation = output["Explanation"]
            frame = self.answerer.video.get_frame_from_second(goto)
            curr_text = f"Time {goto}/{self.answerer.video.length_secs}"
            self.curr = curr_text
            return goto, frame, questions, explanation
        except Exception as e:
            print(e)
    
    def forward(self, video: VideoObj, plan: list[str]) -> (float, PIL.Image.Image, list[str]):
        output = self.select_frame(video, plan)
        goto, frame, questions, explanation = self.parse_answer(output)
        return goto, frame, questions, explanation
