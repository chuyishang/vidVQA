import torch
import openai
import abc
import torch.nn.functional as F
# from lavis.models import load_model_and_preprocess

with open('api.key') as f:
    openai.api_key = f.read().strip()
with open('api_org.key') as f:
    openai.organization = f.read().strip()

device = "cuda" if torch.cuda.is_available() else "cpu"

# ========================== Base abstract model ========================== #
class BaseModel(abc.ABC):
    to_batch = False
    seconds_collect_data = 1.5  # Window of seconds to group inputs, if to_batch is True
    max_batch_size = 10  # Maximum batch size, if to_batch is True. Maximum allowed by OpenAI
    requires_gpu = True

    def __init__(self, gpu_number):
        self.dev = f'cuda:{gpu_number}' if device == 'cuda' else device

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        """
        If to_batch is True, every arg and kwarg will be a list of inputs, and the output should be a list of outputs.
        The way it is implemented in the background, if inputs with defaults are not specified, they will take the
        default value, but still be given as a list to the forward method.
        """
        pass

    @classmethod
    @abc.abstractmethod
    def name(cls) -> str:
        """The name of the model has to be given by the subclass"""
        pass

    @classmethod
    def list_processes(cls):
        """
        A single model can be run in multiple processes, for example if there are different tasks to be done with it.
        If multiple processes are used, override this method to return a list of strings.
        Remember the @classmethod decorator.
        If we specify a list of processes, the self.forward() method has to have a "process_name" parameter that gets
        automatically passed in.
        See GPT3Model for an example.
        """
        return [cls.name]

# ========================== Specific Models ========================== #

class BLIPModel(BaseModel):
    """Model implementation for BLIP-2."""
    name = 'blip-2'
    to_batch = True
    max_batch_size = 32
    seconds_collect_data = 0.2
    # TODO: Create config YAML file
    def __init__(self, gpu_number=0, half_precision=False, 
                 blip_v2_model_type='blip2-flan-t5-xl'):
        super().__init__(gpu_number)

        assert blip_v2_model_type in ['blip2-flan-t5-xxl', 'blip2-flan-t5-xl', 'blip2-opt-2.7b', 'blip2-opt-6.7b',
                                      'blip2-opt-2.7b-coco', 'blip2-flan-t5-xl-coco', 'blip2-opt-6.7b-coco']
        
        #from lavis.models import load_model_and_preprocess
        
        """Imports a processor and BLIP-2 model from HuggingFace. 
        A Blip2Processor prepares images for the model and decodes the predicted tokens ID's back to text.
        """
        from transformers import Blip2ForConditionalGeneration, Blip2Processor
        

        with torch.cuda.device(self.dev):
            max_memory = {gpu_number: torch.cuda.mem_get_info(self.dev)[0]}
            self.processor = Blip2Processor.from_pretrained(f"Salesforce/{blip_v2_model_type}")
            try:
                self.model = Blip2ForConditionalGeneration.from_pretrained(
                    f"Salesforce/{blip_v2_model_type}", load_in_8bit=half_precision,
                    torch_dtype=torch.float16 if half_precision else "auto",
                    device_map="sequential", max_memory=max_memory
                )
            except Exception as e:
                if "had weights offloaded to the disk" in e.args[0]:
                    extra_text = ' You may want to consider setting half_precision to True.' if half_precision else ''
                    raise MemoryError(f"Not enough GPU memory in GPU {self.dev} to load the model.{extra_text}")
                else:
                    raise e
        self.qa_prompt = "Question: {} Short answer:"
        self.caption_prompt = "a photo of"
        self.half_precision = half_precision
        self.max_words = 50
        
    @torch.no_grad()
    def caption(self, image, prompt=None):
        inputs = self.processor(images=image, text=prompt, return_tensors="pt", padding="longest").to(self.dev)
        if self.half_precision:
            inputs['pixel_values'] = inputs['pixel_values'].half()
        generated_ids = self.model.generate(**inputs, length_penalty=-1, num_beams=5, max_length=10, min_length=1,
                                            do_sample=False, top_p=0.9, repetition_penalty=1.0,
                                            num_return_sequences=1, temperature=1)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return generated_text

    @torch.no_grad()
    def qa(self, image, question):
        inputs = self.processor(images=image, text=question, return_tensors="pt", padding="longest").to(self.dev)
        if self.half_precision:
            inputs['pixel_values'] = inputs['pixel_values'].half()
        generated_ids = self.model.generate(**inputs, length_penalty=-1, num_beams=5, max_length=10, min_length=1,
                                            do_sample=False, top_p=0.9, repetition_penalty=1.0,
                                            num_return_sequences=1, temperature=1)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return generated_text

    def forward(self, image, question=None, task='caption'):
        if not self.to_batch:
            image, question, task = [image], [question], [task]
        if len(image) > 0 and 'float' in str(image[0].dtype) and image[0].max() <= 1:
            image = [im * 255 for im in image]
    
        # Separate into qa and caption batches.
        prompts_qa = [self.qa_prompt.format(self.pre_question(q)) for q, t in zip(question, task) if t == 'qa']
        images_qa = [im for i, im in enumerate(image) if task[i] == 'qa']
        images_caption = [im for i, im in enumerate(image) if task[i] == 'caption']
        with torch.cuda.device(self.dev):
            response_qa = self.qa(images_qa, prompts_qa) if len(images_qa) > 0 else []
            response_caption = self.caption(images_caption) if len(images_caption) > 0 else []
        response = []
        for t in task:
            if t == 'qa':
                response.append(response_qa.pop(0))
            else:
                response.append(response_caption.pop(0))
        if not self.to_batch:
            response = response[0]
        return response

class SiglipModel(BaseModel):
    name = "siglip"
    to_batch = True
    max_batch_size = 32
    seconds_collect_data = 0.2
    """Model implementation for SIGLIP."""
    def __init__(self, gpu_number=0, siglip_model_type="ViT-SO400M-14-SigLIP-384"):
        super().__init__(gpu_number)
        with torch.cuda.device(self.dev):
            try:
                from open_clip import create_model_from_pretrained, get_tokenizer
                self.model, self.preprocess = create_model_from_pretrained(f"hf-hub:timm/{siglip_model_type}")
                self.model = self.model.to(self.dev)
                self.tokenizer = get_tokenizer(f"hf-hub:timm/{siglip_model_type}")
            except Exception as e:
                raise Exception(f"Could not load SIGLIP model: {e}")
    
    def prepare_images(self, images):
        image_stack = torch.stack([self.preprocess(image) for image in images])
        return image_stack
    
    def prepare_texts(self, texts):
        text_stack = self.tokenizer(texts, context_length=self.model.context_length)
        return text_stack

    @torch.no_grad()
    def forward(self, images, queries=NotImplementedError, top_k=1):
        if not self.to_batch:
            image, text = [image], [text]
        image_stack = self.prepare_images(images).to(self.dev)
        text_stack = self.prepare_texts(queries).to(self.dev)
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = self.model.encode_image(image_stack)
            text_features = self.model.encode_text(text_stack)
            image_features = F.normalize(image_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)
            #print("Image features shape: ", image_features.shape, "Text features shape: ", text_features.shape)
            text_probs = torch.sigmoid(text_features @ image_features.T * self.model.logit_scale.exp() + self.model.logit_bias)
        # indices returns a matrix of shape [len(queries), top_k], where each row is the top_k indices for that text
        values, indices = torch.topk(text_probs, top_k)
        # TODO: implement functionality for multiple text prompts (batched)
        
        raw_images = []
        for i in range(len(queries)):
            #raw_images.append([indices[i][idx] for idx in range(3)])
            #indices = [indices[i][idx] for idx in range(top_k)] 
            raw_images.append([images[num] for num in [indices[i][idx].item() for idx in range(top_k)]])
        return raw_images

        
class GPTModel(BaseModel):
    """Model implementation for GPT."""
    pass