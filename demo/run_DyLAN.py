import random
from prettytable import PrettyTable
from LLMLP import LLMLP
from utils import *

EXP_NAME = "trial_1"
MODEL = "chatgpt0301"

TYPE = "open-ended"
ACTIVATION = "listwise"  # 推理时智能体选择
QUERY = r"""1+1=?"""
ROLES = ["Assistant", "Mathematician", "Programmer", "Lawyer"]


def set_rd_seed(seed):
    random.seed(seed)


def main():
    set_rd_seed(0)
    assert len(ROLES) > 0

    llmlp = LLMLP(MODEL, len(ROLES), ROLES, 6, ACTIVATION, TYPE, MODEL)

    llmlp.zero_grad()
    res, resp_cnt, completions, prompt_tokens, completion_tokens = llmlp.forward(QUERY)
    # ???
    imp_score = llmlp.backward(res)
    imp_score = [
        [imp_score[idx] for idx in range(len(ROLES) * rid, len(ROLES) * (rid + 1))]
        for rid in range(3)
    ]

    pt = PrettyTable()
    pt.add_column("Round", ROLES)
    for rid in range(3):
        responses = [
            (
                completions[idx][rid]
                if completions[idx][rid] is not None
                else "No response."
            )
            for idx in range(len(ROLES))
        ]
        pt.add_column(str(rid + 1), responses, "l")

    # print(r"Query: {}".format(QUERY))
    # print(r"#API calls: {}".format(resp_cnt))
    # print(r"Prompt Tokens: {}".format(prompt_tokens))
    # print(r"Completion Tokens: {}".format(completion_tokens))
    # print(pt)
    print(r"Final Answer: {}".format(res))
    # ROLES中代理的分数
    print(
        r"Agent Importance Scores: {}".format(
            [sum(imp_score[rid][idx] for rid in range(3)) for idx in range(len(ROLES))]
        )
    )


if __name__ == "__main__":
    main()
