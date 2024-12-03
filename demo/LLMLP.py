import math
import random
from LLM_Neuron import LLMNeuron, LLMEdge, listwise_ranker_2
from utils import parse_single_choice, most_frequent, is_equiv, extract_math_answer
from sacrebleu import sentence_bleu
from prompt_lib import GEN_THRESHOLD


ACTIVATION_MAP = {
    "listwise": 0,
    "trueskill": 1,
    "window": 2,
    "none": -1,
}  # TODO: only 0 is implemented


class LLMLP:

    def __init__(
        self,
        default_model_name,
        agents=4,
        agent_roles=[],
        rounds=2,
        activation="listwise",
        qtype="single_choice",
        mtype="gpt-3.5-turbo",
    ):
        self.default_model_name = default_model_name
        self.agents = agents
        self.rounds = rounds
        self.activation = ACTIVATION_MAP[activation]
        self.mtype = mtype

        assert len(agent_roles) == agents and agents > 0
        self.agent_roles = agent_roles
        self.qtype = qtype
        if qtype == "single_choice":
            self.cmp_res = lambda x, y: x == y
            self.ans_parser = parse_single_choice
        elif qtype == "math_exp":
            self.cmp_res = is_equiv
            self.ans_parser = extract_math_answer
        elif qtype == "open-ended":
            self.cmp_res = (
                lambda x, y: sentence_bleu(x, [y], lowercase=True).score
                >= GEN_THRESHOLD * 100
            )
            self.ans_parser = lambda x: x
        else:
            raise NotImplementedError("Error qtype")

        self.init_nn(self.activation, self.agent_roles)

    def init_nn(self, activation, agent_roles):
        self.nodes, self.edges = [], []
        for idx in range(self.agents):
            self.nodes.append(
                LLMNeuron(agent_roles[idx], self.mtype, self.ans_parser, self.qtype)
            )

        agents_last_round = self.nodes[: self.agents]
        for rid in range(1, self.rounds):
            for idx in range(self.agents):
                self.nodes.append(
                    LLMNeuron(agent_roles[idx], self.mtype, self.ans_parser, self.qtype)
                )
                for a1 in agents_last_round:
                    self.edges.append(LLMEdge(a1, self.nodes[-1]))
            agents_last_round = self.nodes[-self.agents :]

        if activation == 0:
            self.activation = listwise_ranker_2
            self.activation_cost = 1
        else:
            raise NotImplementedError("Error init activation func")

    def zero_grad(self):
        for edge in self.edges:
            edge.zero_weight()

    def check_consensus(self, idxs, idx_mask):
        # check consensus based on idxs (range) and idx_mask (actual members, might exceed the range)
        candidates = [self.nodes[idx].get_answer() for idx in idxs]
        # 找到出现次数最多的答案作为共识答案
        consensus_answer, ca_cnt = most_frequent(candidates, self.cmp_res)
        if ca_cnt > math.floor(2 / 3 * len(idx_mask)):
            return True, consensus_answer
        return False, None

    def set_allnodes_deactivated(self):
        for node in self.nodes:
            node.deactivate()

    def forward(self, question):
        # 获取每个节点的答案
        def get_completions():
            completions = [[] for _ in range(self.agents)]
            for rid in range(self.rounds):
                for idx in range(self.agents * rid, self.agents * (rid + 1)):
                    if self.nodes[idx].active:
                        completions[idx % self.agents].append(
                            self.nodes[idx].get_reply()
                        )
                    else:
                        completions[idx % self.agents].append(None)
            return completions

        resp_cnt = 0
        total_prompt_tokens, total_completion_tokens = 0, 0
        self.set_allnodes_deactivated()
        assert self.rounds > 2

        # loop_indices 每一轮代理的索引
        loop_indices = list(range(self.agents))
        random.shuffle(loop_indices)
        # 第一轮辩论
        activated_indices = []
        for idx, node_idx in enumerate(loop_indices):
            # activate: 包括构造上下文和获取llm回复，并根据对前驱节点的回复评分设置边的权重
            self.nodes[node_idx].activate(question)
            resp_cnt += 1
            print(resp_cnt)
            print
            total_prompt_tokens += self.nodes[node_idx].prompt_tokens
            total_completion_tokens += self.nodes[node_idx].completion_tokens
            activated_indices.append(node_idx)
            # 达到2/3已回复则检查共识
            if idx >= math.floor(2 / 3 * self.agents):
                reached, reply = self.check_consensus(
                    activated_indices, list(range(self.agents))
                )
                if reached:
                    return (
                        reply,
                        resp_cnt,
                        get_completions(),
                        total_prompt_tokens,
                        total_completion_tokens,
                    )

        loop_indices = list(range(self.agents, self.agents * 2))
        random.shuffle(loop_indices)
        # 第二轮辩论 loop_indices --> 345
        activated_indices = []
        for idx, node_idx in enumerate(loop_indices):
            self.nodes[node_idx].activate(question)
            resp_cnt += 1
            total_prompt_tokens += self.nodes[node_idx].prompt_tokens
            total_completion_tokens += self.nodes[node_idx].completion_tokens
            activated_indices.append(node_idx)
            # 检查共识
            if idx >= math.floor(2 / 3 * self.agents):
                reached, reply = self.check_consensus(
                    activated_indices, list(range(self.agents))
                )
                if reached:
                    return (
                        reply,
                        resp_cnt,
                        get_completions(),
                        total_prompt_tokens,
                        total_completion_tokens,
                    )

        # 第三轮及之后的辩论
        idx_mask = list(range(self.agents))  # [0,1,2,3]
        idxs = list(range(self.agents, self.agents * 2))  # 保存上一轮的代理索引
        for rid in range(2, self.rounds):  #
            # 第二轮之后，如果节点数量>=4，则选择表现最好的2个代理
            if self.agents >= 4:
                replies = [self.nodes[idx].get_reply() for idx in idxs]
                indices = list(range(len(replies)))  # 回复长度[0,1,2,3]
                random.shuffle(indices)  # [2,0,3,1]
                shuffled_replies = [replies[idx] for idx in indices]
                # tops 获取表现最好的两个代理 [0,1]
                tops, prompt_tokens, completion_tokens = self.activation(
                    shuffled_replies, question, self.qtype, self.mtype
                )
                total_prompt_tokens += prompt_tokens
                total_completion_tokens += completion_tokens
                # indices[x]回复最好的代理之一的局部索引 idxs[]获取该代理全局索引 -- 取模获取局部索引
                idx_mask = list(
                    map(lambda x: idxs[indices[x]] % self.agents, tops)
                )  # [2,0]
                resp_cnt += self.activation_cost
            # loop_indices 每一轮代理的索引 [8,9,10,11]
            loop_indices = list(range(self.agents * rid, self.agents * (rid + 1)))
            random.shuffle(loop_indices)
            idxs = []
            for idx, node_idx in enumerate(loop_indices):
                if idx in idx_mask:
                    self.nodes[node_idx].activate(question)
                    resp_cnt += 1
                    total_prompt_tokens += self.nodes[node_idx].prompt_tokens
                    total_completion_tokens += self.nodes[node_idx].completion_tokens
                    idxs.append(node_idx)
                    if len(idxs) > math.floor(2 / 3 * len(idx_mask)):
                        reached, reply = self.check_consensus(idxs, idx_mask)
                        if reached:
                            return (
                                reply,
                                resp_cnt,
                                get_completions(),
                                total_prompt_tokens,
                                total_completion_tokens,
                            )

        completions = get_completions()
        return (
            most_frequent([self.nodes[idx].get_answer() for idx in idxs], self.cmp_res)[
                0
            ],
            resp_cnt,
            completions,
            total_prompt_tokens,
            total_completion_tokens,
        )

    # 逆序遍历，计算每个代理对最终答案的贡献，即代理重要性分数
    def backward(self, result):
        flag_last = False
        # 从最后一轮开始
        for rid in range(self.rounds - 1, -1, -1):
            if not flag_last:
                if (
                    len(
                        [
                            idx
                            for idx in range(self.agents * rid, self.agents * (rid + 1))
                            if self.nodes[idx].active
                        ]
                    )
                    > 0
                ):
                    flag_last = True
                else:
                    continue
                # 将当前轮中与最终答案一致的代理的权重之和除以这些代理的数量
                active_indices = [
                    idx
                    for idx in range(self.agents * rid, self.agents * (rid + 1))
                    if self.nodes[idx].active
                    and self.cmp_res(self.nodes[idx].get_answer(), result)
                ]
                ave_w = 1 / len(active_indices)
                # 若该代理是活跃的且其答案与最终答案一致则设置权重为ave_w
                for idx in range(self.agents * rid, self.agents * (rid + 1)):
                    if self.nodes[idx].active and self.cmp_res(
                        self.nodes[idx].get_answer(), result
                    ):
                        self.nodes[idx].importance = ave_w
                    # 否则权重为0
                    else:
                        self.nodes[idx].importance = 0
            else:
                for idx in range(self.agents * rid, self.agents * (rid + 1)):
                    self.nodes[idx].importance = 0
                    if self.nodes[idx].active:
                        for edge in self.nodes[idx].to_edges:
                            self.nodes[idx].importance += (
                                edge.weight * edge.a2.importance
                            )

        return [node.importance for node in self.nodes]
