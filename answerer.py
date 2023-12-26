import modules

class answerer():
    def __init__(self, caption_model, vqa_model, llm):
        self.caption_model = caption_model
        self.vqa_model = vqa_model
        self.llm = llm

    @staticmethod
    def construct_prompt(question, answer_choices, curr_frame, total_frames, caption):
        pass
    

    def get_caption():
        pass

    def query_caption():
        pass

    def select_frame():
        pass

    def get_answer():
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