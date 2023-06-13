from llama_cpp import Llama
import json

if __name__ == "__main__":
    with open("scripts/questions.json","r") as file:
        data = [json.loads(item) for item in file.readlines()]

    llm = Llama(model_path="/home/disk/wxu/finetune_qasum/checkpoint-12/ggml-model-f16.bin",n_ctx=1024,n_gpu_layers=40)
    for line in data:
        prompt = "###Human:{}###Assistant:".format(line["question"])
        output = llm(prompt, max_tokens=400, stop=["###"], echo=True, top_p=0.8, repeat_penalty=1.02, temperature = 0.8)
        line["mit_13B"] = output["choices"][0]["text"][len(prompt):]
        print(line["mit_13B"])

    with open("scripts/qasum_12.json","w") as file:
        json.dump(data,file)
        
