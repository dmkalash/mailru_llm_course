import transformers


class GPTWrapper:

    def __init__(self):
        self.model = transformers.GPT2LMHeadModel.from_pretrained("/home/artyom/gpt_hw/checkpoint3000")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained("ai-forever/ruGPT-3.5-13B")

    def generate(self, input_text, **generation_kwargs):
        inputs = self.tokenizer(input_text, return_tensors='pt')
        inputs.update(generation_kwargs)
        generated_tokens = self.model.generate(**inputs)
        return self.tokenizer.decode(generated_tokens[0])
        

def construct_model():
    generation_kwargs = {
        "max_new_tokens": 40,
        "num_beams": 2,
        "early_stopping": True,
        "no_repeat_ngram_size": 2
    }
    model = GPTWrapper()
    return model, generation_kwargs