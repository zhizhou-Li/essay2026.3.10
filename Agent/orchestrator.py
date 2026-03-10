# -*- coding: utf-8 -*-
# 文件路径: SymbolGeneration/Agent/orchestrator.py
from __future__ import annotations
import os
import requests
from pathlib import Path
from typing import Optional, Union, Dict, Any, List, Tuple

# --- Agents ---
from .agents.interpreter_agent import run_interpreter
from .agents.detector_agent import run_detector
from .agents.extractor_agent import run_extractor
from .agents.designer_agent import run_designer, refine_designer
from .agents.generator_agent import run_generator
from .agents.reviewer_agent import run_reviewer

from .agents.grounder_agent import ground_entity_to_spec, _search_baidu_image  # <--- 引入百度搜图
from .agents.spec_utils import merge_specs, normalize_spec
from .agents.spec_infer_agent import infer_structure_spec
from .agents.vectorizer_agent import png_to_svg
from .agents.photo_symbol_agent import photo_to_symbol
from .config import TARGETS


def pass_threshold(r: dict) -> bool:
    return (
            r.get("clarity_score", 0) >= TARGETS["clarity"]
            and r.get("aesthetic_score", 0) >= TARGETS["aesthetic"]
            and r.get("recognizability_score", 0) >= TARGETS["recognizability"]
            and r.get("structure_penalty", 0) <= 20
    )


def _is_bridge(user_text: str, *specs) -> bool:
    txt = (user_text or "").lower()
    if ("桥" in txt) or ("bridge" in txt):
        return True
    for s in specs:
        if isinstance(s, dict) and (s.get("entity_type") == "bridge"):
            return True
    return False


# [修复版] 下载函数：必须加 Referer
def _download_temp_image(url: str) -> Optional[str]:
    print(f"⬇️ 正在下载参考图: {url[:50]}...")
    try:
        from .utils import BASE_DIR
        out_dir = BASE_DIR / "outputs" / "temp_downloads"
        out_dir.mkdir(parents=True, exist_ok=True)

        fname = "auto_ref_" + str(hash(url))[:8] + ".jpg"
        local_path = out_dir / fname

        # [核心] 破解百度防盗链的 Headers
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Referer": "https://image.baidu.com/"
        }

        # 增加 verify=False 可选，防止 SSL 报错
        resp = requests.get(url, headers=headers, timeout=15)

        if resp.status_code == 200:
            if len(resp.content) < 1000:
                print("⚠️ 下载图片过小，可能是防盗链占位图")
                return None
            local_path.write_bytes(resp.content)
            print(f"✅ 参考图下载完成: {local_path}")
            return str(local_path)
        else:
            print(f"⚠️ 下载失败，状态码: {resp.status_code}")
    except Exception as e:
        print(f"⚠️ 下载异常: {e}")
    return None


def run_micromap_experiment(
        image_path: Optional[str],
        user_text: str,
        user_structure_spec: Optional[Union[Dict[str, Any], str]] = None,
        max_rounds: int = 3,
        force_entity_type: Optional[str] = None,
) -> Dict[str, Any]:
    print("\n🚀 启动 Multi-Agent MicroMap-Agent 实验流程")
    print("📌 文本描述:", user_text)
    if image_path:
        print("📷 引用参考图像:", image_path)

    # 1. Interpreter
    schema = run_interpreter(user_text)
    print("✅ Interpreter 完成")

    # [新增] 从 Schema 中提取精准的地标名称
    target_landmark_name = None
    try:
        import json
        if schema:
            schema_data = json.loads(schema)
            # 获取 entity.name (例如 "兰州白塔山")
            target_landmark_name = schema_data.get("entity", {}).get("name")
            print(f"🎯 提取到精准地标名称: {target_landmark_name}")
    except Exception:
        pass

    # 2. Detector (如果一开始就有图)
    detector_spec: Optional[str] = None
    outline_path: Optional[str] = None
    if image_path:
        try:
            detector_spec = run_detector(image_path, schema)
            print("✅ Detector 完成")
        except Exception as e:
            print(f"⚠️ Detector 失败: {e}")
        try:
            outline_path = run_extractor(image_path)
            print(f"✅ Outline 提取完成")
        except Exception as e:
            print(f"⚠️ Outline 提取失败: {e}")

    # 3. Grounder
    grounder_spec: Optional[Dict[str, Any]] = None
    try:
        # [关键修改] 把提取到的名字传进去！
        grounder_spec = ground_entity_to_spec(user_text, search_focus=target_landmark_name)
        print("✅ Grounder 完成")
    except Exception as e:
        print(f"⚠️ Grounder 失败: {e}")
        grounder_spec = None

    # === 自主视觉检索增强 ===
    if not image_path:
        auto_url = None
        if grounder_spec and grounder_spec.get("reference_image_url"):
            auto_url = grounder_spec["reference_image_url"]
            print(f"🤖 [Auto-Visual] Grounder 提供了参考图链接")
        else:
            # [关键修改] 兜底时也用精准名字搜！
            search_query = target_landmark_name if target_landmark_name else user_text
            print(f"🔎 [Auto-Visual] 尝试自主搜图 (关键词: {search_query})...")
            try:
                auto_url = _search_baidu_image(search_query)
            except Exception as e:
                print(f"⚠️ 兜底搜图失败: {e}")

        # 如果拿到了 URL，就下载并强行注入 image_path
        if auto_url:
            downloaded_path = _download_temp_image(auto_url)
            if downloaded_path:
                image_path = downloaded_path
                print(f"📷 视觉参考已就绪: {image_path}")

                # 补跑 Detector
                if not detector_spec:
                    print("🕵️ 启动 Detector (基于自动检索图)...")
                    try:
                        detector_spec = run_detector(image_path, schema)
                        print("✅ Detector 补跑完成")
                    except Exception as e:
                        print(f"⚠️ Detector 补跑失败: {e}")

                # 补跑 Outline
                if not outline_path:
                    try:
                        outline_path = run_extractor(image_path)
                        print(f"✅ Outline 补跑完成")
                    except Exception as e:
                        print(f"⚠️ Outline 补跑失败: {e}")

    if not image_path:
        print("⚠️ 警告: 未能获取参考图，系统将仅依赖文本生成")
    # ==========================================

    # 4. SpecInfer
    infer_spec: Optional[Dict[str, Any]] = None
    try:
        infer_spec = infer_structure_spec(user_text, detector_spec)
        print("✅ SpecInfer 完成")
    except Exception as e:
        print(f"⚠️ SpecInfer 失败: {e}")
        infer_spec = None

    # 5. Merge Specs
    merged: Dict[str, Any] = {}
    if grounder_spec: merged = merge_specs(defaults=grounder_spec)
    if infer_spec: merged = merge_specs(user_spec=merged, detector_spec=infer_spec)
    if detector_spec: merged = merge_specs(user_spec=merged, detector_spec=detector_spec)
    if user_structure_spec: merged = merge_specs(user_spec=user_structure_spec, detector_spec=merged)

    structure_spec = normalize_spec(merged or {})
    if force_entity_type: structure_spec["entity_type"] = force_entity_type
    is_bridge = _is_bridge(user_text, structure_spec, infer_spec, grounder_spec)

    print("📐 最终结构约束:", structure_spec)

    # 6. Designer
    landmark_json = detector_spec or schema
    style_json = run_designer(landmark_json=landmark_json, schema=schema, structure_spec=structure_spec)
    print("🎨 初始样式 JSON 已生成")

    # 7. Generator Loop
    history: List[Dict[str, Any]] = []
    best_png: Optional[str] = None
    best_review: Optional[Dict[str, Any]] = None
    best_svg: Optional[str] = None

    for round_id in range(1, max_rounds + 1):
        print(f"\n===== 🌀 Round {round_id} / {max_rounds} =====")
        candidate_paths = run_generator(
            outline_path=outline_path,
            style_json=style_json,
            user_text=user_text,
            structure_spec=structure_spec,
        )
        candidate_paths = [p for p in candidate_paths if isinstance(p, str) and not p.startswith("http")]

        if not candidate_paths:
            print("⚠️ 本轮未生成候选图片，终止循环。")
            break

        scored: List[Tuple[str, Dict[str, Any]]] = []
        for path in candidate_paths:
            review = run_reviewer(path, structure_spec=structure_spec)
            scored.append((path, review))

        def total_score(r: Dict[str, Any]) -> float:
            return (
                    float(r.get("clarity_score", 0)) +
                    float(r.get("aesthetic_score", 0)) +
                    float(r.get("recognizability_score", 0)) -
                    0.5 * float(r.get("structure_penalty", 0))
            )

        best_path, round_best_review = max(scored, key=lambda x: total_score(x[1]))
        print("⭐ 本轮最佳:", best_path)
        print("   分数:", {k: round_best_review.get(k) for k in
                           ["clarity_score", "aesthetic_score", "recognizability_score", "structure_penalty"]})

        history.append({
            "round": round_id,
            "candidates": [{"png": p, "review": r} for (p, r) in scored],
            "best_png": best_path,
            "best_review": round_best_review,
        })
        best_png = best_path
        best_review = round_best_review

        if pass_threshold(round_best_review):
            print("✅ 达到目标阈值，提前收敛。")
            break

        if round_id < max_rounds:
            print("🔁 未达标，调用 refine_designer 调整样式 JSON")
            style_json = refine_designer(prev_style_json=style_json, review_data=round_best_review,
                                         structure_spec=structure_spec)
        else:
            print("⏹ 已到最大轮数，停止迭代。")

    # 8. Vectorizer
    if best_png:
        try:
            best_svg = png_to_svg(input_png=best_png, out_svg=None, method="auto", threshold=180, simplify_eps=1.0)
            print(f"✅ 矢量化完成: {best_svg}")
        except Exception as e:
            print(f"⚠️ SVG 矢量化失败: {e}")

    print("\n✅ 实验结束。所有输出已在 Agent/outputs 下生成。")

    return {
        "user_text": user_text,
        "image_path": image_path,
        "best_png": best_png,
        "best_svg": best_svg,
    }


if __name__ == "__main__":
    run_micromap_experiment(
        image_path=None,
        user_text="生成具有艺术化风格的兰州黄河母亲塑像图标，要求结构可辨、黑白二值化、留白均衡",
        user_structure_spec=None,
        max_rounds=1,
        force_entity_type=None,
    )