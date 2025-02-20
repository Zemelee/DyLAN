# 此代码来自其他论文：https://composable-models.github.io/llm_debate
# 仅做复现学习使用
import openai
import numpy as np
import pickle
from tqdm import tqdm

def generate_answer(answer_context):
    key = ""
    baseurl = ""
    model = ""
    client = openai.OpenAI(api_key=key, base_url=baseurl)
    try:
        completion = client.chat.completions.create(
            model=model, messages=answer_context, n=1
        )
    except:
        print("retrying due to an error......")
        return generate_answer(answer_context)
    return completion


# 收集到其他智能体的回答后，构建新的用户提示词，让智能体根据这些信息进行更新答案
def construct_message(agents, question, idx):
    if len(agents) == 0:
        return {
            "role": "user",
            "content": "Can you verify that your answer is correct. Please reiterate your answer, making sure to state your answer at the end of the response.",
        }

    prefix_string = "These are the recent/updated opinions from other agents: "

    for agent in agents:
        agent_response = agent[idx]["content"]
        response = "\n\n One agent response: ```{}```".format(agent_response)

        prefix_string = prefix_string + response

    prefix_string = (
        prefix_string
        + "\n\n Use these opinions carefully as additional advice, can you provide an updated answer? Make sure to state your answer at the end of the response.".format(
            question
        )
    )
    return {"role": "user", "content": prefix_string}


def construct_assistant_message(completion):
    content = completion.choices[0].message.content
    return {"role": "assistant", "content": content}


# 从后往前解析含数字的答案 **246**. 解析错误！
def parse_answer(sentence):
    parts = sentence.split(" ")
    for part in parts[::-1]:
        try:
            answer = float(part)
            return answer
        except:
            continue


def most_frequent(List):
    counter = 0
    num = List[0]

    for i in List:
        current_frequency = List.count(i)
        if current_frequency > counter:
            counter = current_frequency
            num = i

    return num


if __name__ == "__main__":
    answer = ""
    agents = 2
    rounds = 3
    np.random.seed(0)

    evaluation_round = 1  # 评估轮数
    scores = []  # 存储每一轮的得分，准确率指标
    generated_description = {}
    for round in tqdm(range(evaluation_round)):
        a, b, c, d, e, f = np.random.randint(0, 30, size=6)

        answer = a + b * c + d - e * f
        agent_contexts = [
            [
                {
                    "role": "user",
                    "content": f"""What is the result of {a}+{b}*{c}+{d}-{e}*{f}? Make sure to state your answer at the end of the response.""",
                }
            ]
            for agent in range(agents)
        ]

        content = agent_contexts[0][0]["content"]
        question_prompt = f"We seek to find the result of {a}+{b}*{c}+{d}-{e}*{f}?"
        #  为每一个智能体生成回答历史，并分别保存聊天记录
        for round in range(rounds):
            for i, agent_context in enumerate(agent_contexts):
                # 第一轮之后，每个智能体都会收到其他智能体的回答
                if round != 0:
                    agent_contexts_other = agent_contexts[:i] + agent_contexts[i + 1 :]
                    # 构建新的更新答案的提示词
                    message = construct_message(
                        agent_contexts_other, question_prompt, 2 * round - 1
                    )
                    agent_context.append(message)
                    print("message: ", message)

                completion = generate_answer(agent_context)

                assistant_message = construct_assistant_message(completion)
                agent_context.append(assistant_message)

        text_answers = []
        # agent_contexts包含了每个智能体与用户的对话历史
        for agent_context in agent_contexts:
            # 从最后一轮对话种提取答案
            text_answer = string = agent_context[-1]["content"]
            text_answer = text_answer.replace(",", ".")
            text_answer = parse_answer(text_answer)

            if text_answer is None:
                continue

            text_answers.append(text_answer)

        generated_description[(a, b, c, d, e, f)] = (agent_contexts, answer)

        try:
            text_answer = most_frequent(text_answers)
            if text_answer == answer:
                scores.append(1)
            else:
                scores.append(0)
        except:
            continue

        print("performance:", np.mean(scores), np.std(scores) / (len(scores) ** 0.5))

    # 保存 generated_description 到pickle文件中
    pickle.dump(
        generated_description,
        open("math_agents{}_rounds{}.p".format(agents, rounds), "wb"),
    )
    import pdb

    pdb.set_trace()  # 自动断点
    print(answer)
    print(agent_context)
