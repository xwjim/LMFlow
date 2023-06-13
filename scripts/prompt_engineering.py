from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain import OpenAI
from langchain.chains import SequentialChain
from langchain.chains.base import Chain
from langchain.prompts import PromptTemplate
import os
from langchain.chains import LLMChain
from langchain.llms.llamacpp import LlamaCpp
from langchain.chains.question_answering.refine_prompts import DEFAULT_REFINE_PROMPT  

Assistant_prompt = "The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.{history}###Human:{question}###Assistant:"

Refine_prompt = (
    "The original question is as follows: {question}\n"
    "We have provided an existing answer: {existing_answer}\n"
    "We have the opportunity to refine the existing answer"
    "refine the original answer to better, helpful and detailed."
    "###Assistant:"
)

class ChatBot:
    def __init__(self,model_path):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        self.llm = LlamaCpp(model_path=model_path,n_ctx=1024,n_gpu_layers=40, \
            max_tokens=400,top_p=0.8,repeat_penalty=1.02, \
            temperature = 0.8,stop=["###"],)

        assistant_template = PromptTemplate(input_variables=["question","history"], template=Assistant_prompt)
        self.conversation_chain = LLMChain(
                                llm=self.llm,
                                verbose = True,
                                prompt = assistant_template,
                                output_key="existing_answer"
                            )

        refine_template = PromptTemplate(input_variables=["existing_answer","question"], template=Refine_prompt)
        self.refin_chain = LLMChain(
                                llm=self.llm,
                                verbose = True,
                                prompt = refine_template,
                                output_key="output"
                            )
        self.overall_chain = SequentialChain(
            chains=[self.conversation_chain,self.refin_chain],
            input_variables=["question"],
            # Here we return multiple variables
            output_variables=["output"],
            memory = ConversationBufferMemory(human_prefix="###Human",ai_prefix="###Assistant"),
            verbose=True,
        )

    def run(self,data):
        return self.overall_chain.run(data)

if __name__ == "__main__":
    model_path = "/home/disk/wxu/checkpoint-8000/ggml-model-f16.bin"
    chat = ChatBot(model_path)
    answer = chat.run("北京在哪里?")
    print(answer)
    chat.run("有什么好吃的?")
    print(answer)

    # print(chat.conversation_chain.run("北京在哪里"))
    # print(chat.conversation_chain.run("上面还有需要补充的么"))