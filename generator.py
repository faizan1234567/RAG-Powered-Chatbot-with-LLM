import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline
)
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

def setup_tokenizer(model_name: str):
    """
    setup tokenizer for generator model

    Parameters
    ----------
    model_name: tokenizer name
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer

def load_generator(model_name):

    tokenizer = setup_tokenizer(model_name=model_name)
    # bits and bytes params

    # activate 4bit precision base model
    use_4bit = True

    # Compute dtype for 4-bit base models
    bnb_4bit_compute_dtype = "float16"

    # Quantization type (fp4 or nf4)
    bnb_4bit_quant_type = "nf4"

    # Activate nested quantization for 4-bit base models (double quantization)
    use_nested_quant = False

    # setup quantization config
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

    bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,)

    # Check GPU compatibility with bfloat16
    if compute_dtype == torch.float16 and use_4bit:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print("Your GPU supports bfloat16: accelerate training with bf16=True")
            print("=" * 80)
    
    model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,)
    return model


if __name__ == "__main__":
    model_name = "mistralai/Mistral-7B-Instruct-v0.1"
    tokenizer = setup_tokenizer(model_name)
    model = load_generator(model_name)

    text_generation_pipeline = pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    temperature=0.2,
    repetition_penalty=1.1,
    return_full_text=True,
    max_new_tokens=1000,
    do_sample = True
)
    mistral_llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
    
    # prompt template
    prompt_template = """
    ### [INST]
    Instruction: Answer the question based on your
    knowledge about NUST school of interdisplinary engineering & sciences (SINES):

    {context}

    ### QUESTION:
    {question}

    [/INST]
    """

    # Create prompt from prompt template
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template,
    )

    # Create llm chain
    llm_chain = LLMChain(llm=mistral_llm, prompt=prompt)

    # run the generator
    response = llm_chain.invoke({"context": "", "question": "where is SINES located?"})
    
    print(response["text"])