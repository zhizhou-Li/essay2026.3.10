# 文件：Agent/graph_orchestrator.py
from Agent.core.state import SymbolGenerationState

# 导入你现有的 agents (跟 run_multiagent.py 里的导入一样)
from Agent.agents.interpreter_agent import run_interpreter
from Agent.agents.grounder_agent import ground_entity_to_spec
from Agent.agents.spec_infer_agent import infer_spec
from Agent.agents.designer_agent import run_designer, refine_designer
from Agent.agents.generator_agent import run_generator
from Agent.agents.reviewer_agent import run_reviewer
from Agent.agents.vectorizer_agent import VectorizerAgent
from Agent.config import CREATIVE_SAMPLES
import json


class LandmarkGraphNodes:
    """封装所有图节点"""

    @staticmethod
    def node_cognition(state: SymbolGenerationState) -> dict:
        print("\n--- [Node] 认知与视觉锚定 ---")
        user_input = state["user_input"]

        intent_schema = run_interpreter(user_input)
        try:
            intent_data = json.loads(intent_schema) if isinstance(intent_schema, str) else intent_schema
            entity_name = intent_data.get("entity", {}).get("name", user_input)
        except:
            entity_name = user_input

        grounder_spec = ground_entity_to_spec(user_input, search_focus=entity_name)
        final_spec = infer_spec(user_input, grounder_spec)

        # 提取 VLM Truth 和参考图
        ref_images = grounder_spec.get("reference_images", {})
        ref_url = ref_images.get("isometric") or ref_images.get("front") or grounder_spec.get("reference_image_url")
        vlm_fact_str = ""
        if "vlm_analysis" in grounder_spec:
            v = grounder_spec["vlm_analysis"]
            vlm_fact_str = f"Posture: {v.get('posture')}, Shape: {v.get('shape_description')}"

        # 返回需要更新到全局状态的字段
        return {
            "intent_schema": intent_schema,
            "entity_name": entity_name,
            "grounder_spec": grounder_spec,
            "final_spec": final_spec,
            "ref_url": ref_url,
            "vlm_fact_str": vlm_fact_str
        }

    @staticmethod
    def node_design(state: SymbolGenerationState) -> dict:
        print(f"\n--- [Node] 样式设计 (Round {state['round_idx'] + 1}) ---")
        if state.get("critique"):
            # 如果存在差评，说明是重构图，调用 refine
            style_json = refine_designer(
                prev_style_json=state["current_style_json"],
                review_data={"critique": state["critique"]},  # 简化适配
                structure_spec=state["final_spec"]
            )
        else:
            # 首次设计
            style_json = run_designer(state["entity_name"], state["intent_schema"], state["final_spec"])

        return {"current_style_json": style_json, "round_idx": state["round_idx"] + 1}

    @staticmethod
    def node_generate(state: SymbolGenerationState) -> dict:
        print("\n--- [Node] 候选图像生成 ---")
        paths = run_generator(
            outline_path=None,
            style_json=state["current_style_json"],
            user_text=state["entity_name"],
            structure_spec=state["final_spec"]
        )
        return {"candidate_paths": paths}

    @staticmethod
    def node_review(state: SymbolGenerationState) -> dict:
        print("\n--- [Node] VLM 四维评估 ---")
        best_path = None
        best_acc = -1
        best_review = {}

        # 遍历生成的候选图并打分
        for path in state["candidate_paths"]:
            review = run_reviewer(
                candidate_path=path,
                reference_url=state["ref_url"],
                entity_name=state["entity_name"],
                visual_instruction=state["vlm_fact_str"]
            )
            acc = review.get("scores", {}).get("semantic_accuracy", 0)
            if acc > best_acc:
                best_acc = acc
                best_path = path
                best_review = review

        return {
            "best_candidate_path": best_path,
            "acc_score": best_acc,
            "decision": best_review.get("decision", "FAIL"),
            "critique": best_review.get("critique", "Improve structure.")
        }

    @staticmethod
    def node_vectorize(state: SymbolGenerationState) -> dict:
        print("\n--- [Node] 智能矢量化与拓扑重建 ---")
        # 实例化 Agent 并传入 image_path 和 final_spec (语义意图)
        agent = VectorizerAgent()
        svg_path = agent.run(state["best_candidate_path"], state["final_spec"])
        return {"final_svg_path": svg_path}


class LandmarkGraphWorkflow:
    def __init__(self, max_rounds=3, required_accuracy=8):
        self.max_rounds = max_rounds
        self.required_accuracy = required_accuracy
        self.nodes = LandmarkGraphNodes()

    def run(self, user_input: str) -> SymbolGenerationState:
        # 初始化 State
        state: SymbolGenerationState = {
            "user_input": user_input, "intent_schema": None, "entity_name": "",
            "grounder_spec": None, "final_spec": None, "vlm_fact_str": "", "ref_url": "",
            "current_style_json": None, "round_idx": 0, "candidate_paths": [],
            "best_candidate_path": None, "critique": None, "decision": None, "acc_score": 0,
            "final_svg_path": None
        }

        # 1. 走直线边: Start -> Cognition
        state.update(self.nodes.node_cognition(state))

        # 2. 进入带条件边 (Conditional Edges) 的循环区域
        while True:
            # Graph 边: -> Design
            state.update(self.nodes.node_design(state))

            # Graph 边: Design -> Generate
            state.update(self.nodes.node_generate(state))

            # Graph 边: Generate -> Review
            state.update(self.nodes.node_review(state))

            # Graph 条件边路由 (Conditional Routing)
            print(f"🧐 [Routing] 决策: {state['decision']}, 准确度: {state['acc_score']}")

            if state["decision"] == "PASS" and state["acc_score"] >= self.required_accuracy:
                print("✅ 满足科研标准，跳出生成闭环！")
                break  # 流转到 Vectorize

            if state["round_idx"] >= self.max_rounds:
                print("🛑 达到最大迭代次数，强制结束闭环！")
                break  # 流转到 Vectorize

            print(f"⚠️ 未达标，携带修改意见流转回 Design 节点: {state['critique']}")
            # 如果不 break，循环会自然流转回 node_design

        # 3. 走直线边: -> Vectorize
        state.update(self.nodes.node_vectorize(state))

        print(f"🎉 任务结束！最终输出: {state['final_svg_path']}")
        return state
