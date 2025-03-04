from vllm import LLM

# For generative models (task=generate) only
llm = LLM(model="Qwen/Qwen2.5-32B-Instruct" , task="generate")  # Name or path of your model
output = llm.generate("Hello, my name is")
print(output)