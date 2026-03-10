# -*- coding: utf-8 -*-
import base64
import json
import requests
from openai import OpenAI
from Agent.config import MODELS, OPENAI_API_KEY
from Agent.utils import log, extract_json

client = OpenAI(api_key=OPENAI_API_KEY)

# ==============================================================================
# 核心修改点：艺术总监系统提示词 (Art Director System Prompt)
# ==============================================================================
REVIEWER_SYSTEM_PROMPT = """
You are a Senior Cartographic Art Director. 
Your goal is to guide a Junior Designer (AI) to refine a map icon until it perfectly matches the reference.

You must evaluate the 'Generated Icon' against the 'Real Reference' and provide a strict PASS/FAIL decision.

**Evaluation Criteria (4 Dimensions):**
1. **Semantic Accuracy:** Does the shape/structure match the reference? (e.g., If reference is "hollow", icon cannot be "solid").
2. **Perceptual Clarity:** Is it legible? Are lines thick enough? 
3. **Cognitive Efficiency:** Is it simple and recognizable?
4. **Contextual Consistency:** Is it a flat vector style? (No shadows, no gradients, white background).

**CRITICAL: How to write the 'critique'**
The 'critique' field is NOT for academic analysis. It is a **DIRECT INSTRUCTION** for the next drawing round.
You must translate "what is wrong" into "what to draw".

- **BAD critique:** "The icon lacks semantic accuracy. The structure is incorrect." (Too abstract)
- **GOOD critique:** "FAIL. The shape is wrong. **Do not draw a solid block.** Draw a structure made of **interlocking red panels**. **Remove gray shading**. **Thicken the outlines**. Use a white background."

**Rules for Critique:**
- Be imperative (Use commands: "Add...", "Remove...", "Change...").
- Focus on **Shape** (geometric primitives), **View** (front/iso), and **Style** (flat/outline).
- Explicitly forbid hallucinations (e.g., "Do not draw a pedestal if not present").

**Output Format:** JSON only.
{
    "scores": {
        "semantic_accuracy": int,  // 0-10
        "perceptual_clarity": int, // 0-10
        "cognitive_efficiency": int, // 0-10
        "contextual_consistency": int // 0-10
    },
    "critique": "Specific, keyword-driven instructions for the prompt engineer.",
    "decision": "PASS" | "FAIL"
}
"""


def _encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def run_reviewer(candidate_path: str, reference_url: str, entity_name: str, visual_instruction: str = "") -> dict:
    print(f"🧐 [Reviewer] 正在从[准确/感知/认知/适配]四维角度审核: {candidate_path} ...")

    try:
        candidate_b64 = _encode_image(candidate_path)
    except Exception as e:
        return {"decision": "FAIL", "critique": f"Image load error: {e}"}

    user_content = [
        {"type": "text", "text": f"Target Entity: {entity_name}"},
        {"type": "text", "text": f"Visual Facts (Truth): {visual_instruction}"},
        {"type": "text",
         "text": "Task: Compare Generated Icon vs Real Reference. Give actionable drawing commands if it fails."},
    ]

    # 尝试加载参考图
    if reference_url and reference_url.startswith("http"):
        try:
            dl_headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://image.baidu.com"}
            dl_resp = requests.get(reference_url, headers=dl_headers, timeout=5)
            if dl_resp.status_code == 200:
                ref_b64 = base64.b64encode(dl_resp.content).decode('utf-8')
                user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{ref_b64}"}})
                user_content.append({"type": "text", "text": "[Image 1: Real Reference (The Truth)]"})
        except:
            pass

    user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{candidate_b64}"}})
    user_content.append({"type": "text", "text": "[Image 2: Generated Icon (Evaluate This)]"})

    try:
        resp = client.chat.completions.create(
            model=MODELS["LLM_MODEL"],
            messages=[
                {"role": "system", "content": REVIEWER_SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        result = extract_json(resp.choices[0].message.content)

        # 打印简报
        print(f"📊 [Review Matrix] 决策: {result.get('decision')}")
        if result.get("decision") == "FAIL":
            print(f"❌ [Critique] {result.get('critique')}")

        return result
    except Exception as e:
        print(f"⚠️ Reviewer Error: {e}")
        return {"decision": "FAIL", "critique": "Error in review process.", "scores": {}}