# 文件：Agent/agents/vectorizer_agent.py
import json
from pathlib import Path
from Agent.utils import log, extract_json
from Agent.config import MODELS, OPENAI_API_KEY
from openai import OpenAI

# 导入核心工具函数
from tools.run_color_vectorizer import color_vectorize_pipeline
from agents.semantic_vectorizer import structure_driven_pipeline
from tools.check_topology import check_svg_topology

client = OpenAI(api_key=OPENAI_API_KEY)


class VectorizerAgent:
    def __init__(self, max_retries=3):
        self.max_retries = max_retries

    def run(self, image_path: str, final_spec: dict) -> str:
        log("VectorizerAgent", f"🚀 启动智能矢量化决策引擎...")

        # --- [第一阶段：LLM 自主决策提取路径] ---
        decision = self._decide_routing(image_path, final_spec)
        path_type = decision.get("path", "structure")
        params = decision.get("initial_params", {"epsilon": 1.0, "n_clusters": 4})

        last_svg = None

        # --- [第二阶段：反思与动态调优闭环] ---
        for attempt in range(self.max_retries):
            log("VectorizerAgent", f"🌀 轮次 {attempt + 1}: 执行 {path_type} 路径, 参数: {params}")

            try:
                if path_type == "structure":
                    last_svg = structure_driven_pipeline(image_path, **params)
                else:
                    last_svg = color_vectorize_pipeline(image_path, **params)
            except Exception as e:
                log("VectorizerAgent", f"⚠️ 矢量化执行出错: {e}")
                break  # 严重错误直接跳出

            if not last_svg:
                break

            # --- [第三阶段：统一几何重建与拓扑质检] ---
            topo_report = check_svg_topology(last_svg)
            error_count = topo_report.get("self_intersections", 0)

            if error_count == 0:
                log("VectorizerAgent", "✅ 拓扑检查 100% 通过，数据已达到 Analysis-Ready 标准。")
                break

            # --- [第四阶段：基于 LLM 反馈动态推断超参数] ---
            log("VectorizerAgent", f"⚠️ 检测到 {error_count} 处拓扑异常，呼叫大模型重新推演参数...")
            if attempt < self.max_retries - 1:
                params = self._reflect_and_adjust(params, topo_report, error_count)

        return last_svg

    def _decide_routing(self, image_path: str, spec: dict) -> dict:
        """【真·智能体】调用 LLM 依据语义意图决策提取路径和初始参数"""
        system_prompt = """
        You are an expert GIS Algorithm Engineer. Your task is to select the optimal vectorization pipeline for a map symbol based on its design specification.

        Available pipelines:
        1. "structure": Best for minimalist, black-and-white, highly geometric, or simple line-art icons (e.g., flat bridge silhouettes).
        2. "color": Best for complex, multi-colored, illustrative, or styled symbols.

        Rules:
        - Analyze the user's specification (entity_type, constraints, palette).
        - Output strictly in JSON format.
        - Default epsilon (Douglas-Peucker simplification tolerance) is usually 1.0 to 2.0.
        - If 'color' is chosen, infer a reasonable 'n_clusters' (usually 3 to 6) based on the design's complexity.

        JSON Schema:
        {
            "path": "structure" | "color",
            "initial_params": {
                "epsilon": float,
                "n_clusters": int (only needed if path is color)
            },
            "reason": "brief explanation"
        }
        """

        try:
            resp = client.chat.completions.create(
                model=MODELS["LLM_MODEL"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Design Specification:\n{json.dumps(spec, ensure_ascii=False)}"}
                ],
                response_format={"type": "json_object"},
                temperature=0.1  # 保持较低温度以保证稳定性
            )
            content = resp.choices[0].message.content
            decision = extract_json(content)
            log("VectorizerAgent_Routing", decision)

            # 容错处理
            if not decision or "path" not in decision:
                return {"path": "color", "initial_params": {"n_clusters": 4, "epsilon": 1.0}}
            return decision

        except Exception as e:
            log("VectorizerAgent_Routing_Error", str(e))
            # 兜底：如果 API 失败，退回默认策略
            return {"path": "color", "initial_params": {"n_clusters": 4, "epsilon": 1.0}}

    def _reflect_and_adjust(self, prev_params: dict, report: dict, error_count: int) -> dict:
        """【真·智能体】调用 LLM 依据拓扑错误报告，像人类一样推演并调整参数"""
        system_prompt = """
        You are a Topology Optimization Agent. The previous vectorization attempt yielded topological errors (e.g., self-intersecting polygons).
        Your goal is to adjust the algorithm hyperparameters to eliminate these errors.

        Rules:
        - The primary parameter to fix self-intersections is "epsilon" (Douglas-Peucker tolerance). 
        - Increasing "epsilon" simplifies the geometry, reducing vertices and eliminating complex self-intersections.
        - If the error count is high (>10), make a bolder adjustment (e.g., +1.0 or +1.5).
        - If the error count is low (<3), make a minor tweak (e.g., +0.5).
        - Output strictly in JSON format.

        JSON Schema:
        {
            "epsilon": float (the new updated value),
            "n_clusters": int (keep the previous value if it exists),
            "reason": "Explain why you adjusted the epsilon by this specific amount."
        }
        """

        user_content = f"""
        Previous Parameters: {json.dumps(prev_params)}
        Topology Report: Found {error_count} self-intersecting polygons.
        Please provide the new parameters to fix this.
        """

        try:
            resp = client.chat.completions.create(
                model=MODELS["LLM_MODEL"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                response_format={"type": "json_object"},
                temperature=0.2
            )
            content = resp.choices[0].message.content
            new_params_full = extract_json(content)
            log("VectorizerAgent_Reflect", new_params_full)

            # 提取 LLM 给出的新参数
            new_params = prev_params.copy()
            if "epsilon" in new_params_full:
                new_params["epsilon"] = new_params_full["epsilon"]
            if "n_clusters" in new_params_full and "n_clusters" in new_params:
                new_params["n_clusters"] = new_params_full["n_clusters"]

            return new_params

        except Exception as e:
            log("VectorizerAgent_Reflect_Error", str(e))
            # 兜底：如果 API 失败，执行传统硬编码降级策略
            new_params = prev_params.copy()
            new_params["epsilon"] = new_params.get("epsilon", 1.0) + 0.5
            return new_params


# 兼容旧接口的包装函数
def run_vectorizer_agent(image_path: str, final_spec: dict):
    agent = VectorizerAgent()
    return agent.run(image_path, final_spec)