# -*- coding: utf-8 -*-
# 文件路径: SymbolGeneration/Agent/agents/grounder_agent.py
from __future__ import annotations
import json, re, requests, io
from typing import Dict, Any, Optional, List, Tuple
from bs4 import BeautifulSoup
from openai import OpenAI
from PIL import Image
from sentence_transformers import SentenceTransformer, util
import torch

from ..utils import log, save_json, extract_json
from ..config import OPENAI_API_KEY, MODELS
import base64

# --- Global Models ---
# 加载 CLIP 模型 (Lazy loading is better in production, but global is fine here)
print("⏳ Loading CLIP Model (clip-ViT-B-32)...")
try:
    CLIP_MODEL = SentenceTransformer('clip-ViT-B-32')
    print("✅ CLIP Model Loaded.")
except Exception as e:
    print(f"❌ Failed to load CLIP Model: {e}")
    CLIP_MODEL = None

client = OpenAI(api_key=OPENAI_API_KEY)

# --- Endpoints ---
WIKI_SEARCH = "https://{lang}.wikipedia.org/w/api.php"
WIKI_SUMMARY = "https://{lang}.wikipedia.org/api/rest_v1/page/summary/{title}"


# ==============================================================================
# [Core Engine] Multi-View Visual Reasoning & Semantic Anchoring
# ==============================================================================

def _search_baidu_candidates(keyword: str, limit: int = 30) -> List[str]:
    """
    [Infrastructure] 纯粹的搜图函数，只负责返回 URL 列表，不负责筛选。
    为了多视角选择，我们需要更大的候选池 (limit=30)。
    """
    print(f"🔎 [Baidu Raw] 正在挖掘 '{keyword}' 的视觉素材库 (Limit: {limit})...")
    url = "https://image.baidu.com/search/acjson"

    params = {
        "tn": "resultjson_com", "logid": "8305096434442765369", "ipn": "rj", "ct": "201326592",
        "is": "", "fp": "result", "queryWord": keyword, "cl": "2", "lm": "-1", "ie": "utf-8",
        "oe": "utf-8", "adpicid": "", "st": "-1", "z": "", "ic": "0", "hd": "", "latest": "",
        "copyright": "", "word": keyword, "s": "", "se": "", "tab": "", "width": "", "height": "",
        "face": "0", "istype": "2", "qc": "", "nc": "1", "fr": "", "expermode": "", "force": "",
        "pn": "0", "rn": str(limit), "gsm": "1e",
    }

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "text/plain, */*; q=0.01", "Referer": "https://image.baidu.com/search/index",
        "X-Requested-With": "XMLHttpRequest",
    }

    candidates = []
    try:
        res = requests.get(url, params=params, headers=headers, timeout=6)
        if res.status_code == 200:
            try:
                # Handle potentially malformed JSON strings from Baidu
                json_str = res.text.replace(r"\'", "'")
                data = json.loads(json_str)
                if "data" in data and isinstance(data["data"], list):
                    for item in data["data"]:
                        if not isinstance(item, dict): continue
                        img_url = item.get("thumbURL") or item.get("middleURL")
                        if img_url:
                            candidates.append(img_url)
            except Exception:
                pass
    except Exception as e:
        print(f"⚠️ 搜图网络错误: {e}")

    # 去重并返回
    return list(set(candidates))


def _multi_view_clip_selection(query_text: str, candidate_urls: List[str]) -> Dict[str, str]:
    """
    [Paper Core] Multi-View Semantic Anchoring
    【升级】增加了 'top_down' (俯视) 维度，专门用于解决圆环桥/体育馆等平面几何体被误判的问题。
    """
    if not candidate_urls or CLIP_MODEL is None:
        return {}

    print(f"🧠 [CLIP-3D] 正在构建 {query_text} 的多视角空间认知模型...")

    # ... (中间的图片下载和预处理代码保持不变) ...
    # ... valid_images = [] ...

    # 1. 批量下载 (保持原逻辑，省略重复代码以节省篇幅，请保留原有的下载循环)
    valid_images = []
    valid_urls = []
    headers = {"User-Agent": "Mozilla/5.0"}
    for url in candidate_urls:
        try:
            r = requests.get(url, headers=headers, timeout=2)
            if r.status_code == 200:
                img = Image.open(io.BytesIO(r.content))
                if img.mode != 'RGB': img = img.convert('RGB')
                w, h = img.size
                if w > 150 and h > 150:
                    valid_images.append(img)
                    valid_urls.append(url)
        except:
            continue
        if len(valid_images) >= 25: break  # 稍微增加一点上限

    if not valid_images: return {}

    # --- Step A: 定义多视角 Prompts (【关键修改】：增加 top_down) ---
    view_prompts = {
        "front": f"A direct front view photo of {query_text}, symmetrical facade, elevation",
        "side": f"A side profile view of {query_text}, lateral perspective",
        "isometric": f"An isometric 45-degree view of {query_text}, 3D structure",
        # [NEW] 专门捕捉圆环/平面几何特征
        "top_down": f"A direct top-down aerial map view of {query_text}, satellite image, plan view, complete footprint"
    }

    # ... (负向 Prompts 和 CLIP 计算逻辑保持不变) ...

    # [恢复计算代码]
    negative_texts = ["close-up detail", "partial view", "blurred", "text overlay", "night view", "human selfie"]
    results = {}

    try:
        img_embs = CLIP_MODEL.encode(valid_images)
        neg_embs = CLIP_MODEL.encode(negative_texts)
        neg_sim_matrix = util.cos_sim(img_embs, neg_embs)
        neg_penalties = neg_sim_matrix.max(dim=1).values.numpy()

        for view_name, prompt in view_prompts.items():
            text_emb = CLIP_MODEL.encode([prompt])
            pos_scores = util.cos_sim(text_emb, img_embs)[0].numpy()

            best_score = -999.0
            best_idx = -1

            for i in range(len(valid_images)):
                base_score = pos_scores[i] - (0.35 * neg_penalties[i])
                final_score = base_score

                # [几何约束微调]
                img = valid_images[i]
                w, h = img.size
                ratio = w / h

                if view_name == "front":
                    if ratio < 0.4 or ratio > 2.5: final_score -= 0.15
                elif view_name == "top_down":
                    # 俯视图通常比较方正，或者是横向的
                    if ratio > 0.8: final_score += 0.05

                # ... (其他约束保持不变) ...

                if final_score > best_score:
                    best_score = final_score
                    best_idx = i

            if best_idx != -1 and pos_scores[best_idx] > 0.25:
                results[view_name] = valid_urls[best_idx]
                print(f"✅ [View: {view_name}] Selected (Sim: {pos_scores[best_idx]:.3f})")
            else:
                # 即使有最高分，但分数太低（说明都不相关），宁缺毋滥
                if best_idx != -1:
                    print(f"🗑️ [View: {view_name}] Rejected (Best Sim {pos_scores[best_idx]:.3f} < 0.25)")


        return results

    except Exception as e:
        print(f"⚠️ Multi-view selection error: {e}")
        return {}


def _analyze_image_semantics(image_url: str, entity_name: str) -> Dict[str, str]:
    """
    [Visual Question Answering]
    使用 VLM (GPT-4o) 读取视觉结构。
    【升级】引入 "Knowledge-First" 策略，强制模型用内部知识校准视觉错觉。
    """
    if not image_url: return {}

    print(f"👁️ [VLM] 正在对选中图片进行视觉结构分析 (Base64 Mode): {entity_name} ...")

    # 1. 本地下载图片并转为 Base64 (保持不变)
    base64_image = ""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(image_url, headers=headers, timeout=5)
        if resp.status_code == 200:
            base64_image = base64.b64encode(resp.content).decode('utf-8')
        else:
            print(f"⚠️ 本地下载图片失败: {resp.status_code}")
            return {}
    except Exception as e:
        print(f"⚠️ 图片转Base64失败: {e}")
        return {}

    # 2. 构造 Prompt (【关键修改】)
    # 强制要求 Recall Knowledge (回忆知识) -> Analyze Image (分析图片) -> Reconcile (调和冲突)
    prompt = f"""
    You are an expert cartographer analyzing the landmark '{entity_name}'.

    CRITICAL INSTRUCTION - KNOWLEDGE FIRST:
    1. First, RECALL your internal knowledge about this landmark's true 3D topology. 
       (e.g. Is 'Laguna Garzón Bridge' a ring? Does 'JK Bridge' have 3 arches?)
    2. Then, analyze the image.
    3. DETECT ILLUSIONS: If the image view (e.g. side view) hides the true shape (e.g. makes a ring look like a straight line), 
       you MUST report the TRUE 3D SHAPE based on your knowledge, not just the 2D projection.

    Return JSON only:
    {{
        "posture": "standing|sitting|reclining(lying_down)|abstract|ring_shaped|crossing_arches", 
        "orientation": "vertical(tall)|horizontal(wide)|square",
        "shape_description": "A precise description of the TRUE 3D structural composition, correcting any perspective distortions in the image."
    }}
    """

    # 3. 发送给 OpenAI (保持不变)
    try:
        response = client.chat.completions.create(
            model=MODELS["LLM_MODEL"],
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "low"},
                        },
                    ],
                }
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        content = response.choices[0].message.content
        return extract_json(content) or {}
    except Exception as e:
        print(f"⚠️ VLM Analysis Failed: {e}")
        return {}

# [Compatibility Wrapper] 保持旧接口，供 _gather_raw_knowledge 调用
def _search_baidu_image(keyword: str) -> Optional[str]:
    """
    Legacy wrapper: uses the new candidate search and returns the first result.
    This ensures backward compatibility for functions expecting a single URL.
    """
    cands = _search_baidu_candidates(keyword, limit=5)
    return cands[0] if cands else None


# ==============================================================================
# [Orchestrator] Main Spec Generation Logic
# ==============================================================================

SYSTEM_TO_SPEC = (
    "You are a visual knowledge extraction expert. "
    "Your task is to convert vague user intent and raw encyclopedia snippets into a STRICT visual structure spec.\n"
    "Goal: Extract specific physical constraints so a blind painter can reconstruct the landmark accurately.\n"
    "Schema:\n"
    "{ \n"
    "  \"entity\": {\"name\": str, \"location\": str},\n"
    "  \"entity_type\": \"bridge|tower|building|statue|logogram|other\",\n"
    "  \"structure\": {\n"
    "      \"structural_system\": \"truss|arch|suspension|beam|unknown\",\n"
    "      \"shape_features\": [str],  // e.g. \"3 spans\", \"octagonal base\", \"reclining posture\"\n"
    "      \"material\": \"steel|stone|concrete|wood\",\n"
    "      \"view_recommendation\": \"side|front|isometric\"\n"
    "  },\n"
    "  \"constraints\": {\n"
    "      \"must\": [str],      // Visual elements that MUST appear\n"
    "      \"must_not\": [str]   // Elements to EXCLUDE\n"
    "  }\n"
    "}\n"
    "Rules:\n"
    "1. Rely HEAVILY on the provided snippets.\n"
    "2. If snippets describe a statue, extract posture and composition details.\n"
    "3. Return ONLY a JSON object."
)


def ground_entity_to_spec(user_text: str, search_focus: str = None) -> Dict[str, Any]:
    """
    [Orchestrator] 主控函数
    【升级】实施 "Multi-View Redundancy Retrieval" (多视角冗余检索) 策略。
    """
    # 1. 获取文本知识
    raw_text, ref_img_legacy = _gather_raw_knowledge(user_text, search_focus=search_focus)

    # 2. [关键修改] 生成鲁棒的搜索查询 (Robust Queries)
    # 不再只依赖一个关键词，而是主动去搜特定的视角
    base_query = search_focus if search_focus else user_text

    robust_queries = [
        base_query,  # 基础搜
        f"{base_query} aerial top down view",  # 强制搜俯视 (解决圆环桥问题)
        f"{base_query} isometric structure diagram"  # 强制搜结构 (解决JK大桥问题)
    ]

    print(f"🚀 [Robust Search] 启动多视角检索策略: {robust_queries}")

    # 3. 并发挖掘候选图 (Aggregated Candidates)
    all_candidates = []
    for q in robust_queries:
        # 每个视角搜 15 张，保证池子够大
        cands = _search_baidu_candidates(q, limit=15)
        all_candidates.extend(cands)

    # 去重
    all_candidates = list(set(all_candidates))
    print(f"📦 [Candidate Pool] 共挖掘到 {len(all_candidates)} 张候选视觉素材")

    # 4. 视觉推理与筛选
    visual_pack = _multi_view_clip_selection(base_query, all_candidates)

    # 5. [VLM 分析] 优先顺序调整：如果有俯视图(top_down)，优先分析它
    # 因为对于地图图标来说，俯视图往往包含了最准确的拓扑结构
    target_img_url = (
            visual_pack.get("top_down") or
            visual_pack.get("isometric") or
            visual_pack.get("front") or
            visual_pack.get("side")
    )

    vlm_traits = {}
    if target_img_url:
        vlm_traits = _analyze_image_semantics(target_img_url, user_text)
        print(f"👁️ [VLM Result] 修正后姿态: {vlm_traits.get('posture')} | 描述: {vlm_traits.get('shape_description')}")

    # 6. 构建 Prompt Context
    if not raw_text and not visual_pack and not ref_img_legacy:
        spec = {"entity": {"name": user_text}, "constraints": {"must_not": []}}
        save_json("Grounder_spec", spec)
        return spec

    visual_context_str = "Visual References Retrieved (Multi-View):\n"
    if visual_pack:
        if visual_pack.get("top_down"):
            visual_context_str += f"- Top-Down/Aerial View (Topology): {visual_pack['top_down']}\n"
        if visual_pack.get("isometric"):
            visual_context_str += f"- Isometric View (3D Structure): {visual_pack['isometric']}\n"
        if visual_pack.get("front"):
            visual_context_str += f"- Front View (Facade): {visual_pack['front']}\n"
        if visual_pack.get("side"):
            visual_context_str += f"- Side View (Profile): {visual_pack['side']}\n"
    elif ref_img_legacy:
        visual_context_str += f"- General Reference: {ref_img_legacy}\n"

    if vlm_traits:
        visual_context_str += "\n[VERIFIED VISUAL FACTS via VLM & Knowledge]:\n"
        visual_context_str += f"- True 3D Posture: {str(vlm_traits.get('posture')).upper()}\n"
        visual_context_str += f"- Geometric Composition: {vlm_traits.get('shape_description')}\n"

    msg_user = [
        {"type": "text", "text": f"User intent: {user_text}"},
        {"type": "text", "text": f"Raw encyclopedia snippets:\n{raw_text}"},
        {"type": "text", "text": visual_context_str}
    ]

    # 7. 调用 LLM 生成 Spec (保持不变)
    resp = client.chat.completions.create(
        model=MODELS["LLM_MODEL"],
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": SYSTEM_TO_SPEC}, {"role": "user", "content": msg_user}]
    )
    spec = extract_json(resp.choices[0].message.content) or {"entity": {"name": user_text}}

    if not spec.get("constraints"): spec["constraints"] = {}
    spec["constraints"].setdefault("must_not", [])

    if visual_pack:
        spec["reference_images"] = visual_pack
        # 优先给 Designer 推荐 isometric 或 top_down，因为它们信息量最大
        spec["reference_image_url"] = visual_pack.get("isometric") or visual_pack.get("top_down") or visual_pack.get(
            "front")
    elif ref_img_legacy:
        spec["reference_image_url"] = ref_img_legacy
        spec["reference_images"] = {"front": ref_img_legacy}

    save_json("Grounder_spec", spec)
    return spec


# ==============================================================================
# [Helpers] Encyclopedia & Text Processing
# ==============================================================================

def _gather_raw_knowledge(user_text: str, search_focus: str = None) -> Tuple[str, Optional[str]]:
    queries = _expand_queries(user_text)
    blobs = []
    first_image = None

    if search_focus:
        queries.insert(0, search_focus)

    has_chinese = any('\u4e00' <= ch <= '\u9fff' for ch in user_text)

    for q in queries:
        # 1. 尝试百度百科
        if has_chinese:
            summary, img = _fetch_baidu_baike(q)
            if summary:
                blobs.append(f"[Baidu] {q}\n{summary}")
                if not first_image and img: first_image = img

                # 如果百科没图，这里我们暂时不强制搜图，留给后面的 Multi-View 引擎去做
                continue

        # 2. 尝试维基百科
        langs = _langs_for(q, user_text)
        for lang in langs:
            title = _wiki_search(q, lang)
            if title:
                data = _wiki_summary(title, lang)
                if data:
                    extract = data.get("extract")
                    img_src = data.get("thumbnail", {}).get("source") or data.get("originalimage", {}).get("source")

                    if extract:
                        blobs.append(f"[Wiki-{lang}] {title}\n{extract}")
                        if not first_image and img_src:
                            first_image = img_src
                        break

    # 这里的图片搜索作为最后的最后兜底 (Fallback)
    if not first_image and has_chinese:
        # 只有在完全没有图的时候，才尝试搜一张
        pass
        # 注：为了性能，我们不再这里强制搜图，而是依赖 ground_entity_to_spec 里的 _search_baidu_candidates

    text = "\n\n".join(blobs)
    log("Grounder_raw", text if text else "(empty)")

    return text, first_image


def _fetch_baidu_baike(keyword: str) -> Tuple[Optional[str], Optional[str]]:
    url = f"https://baike.baidu.com/item/{keyword}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    try:
        resp = requests.get(url, headers=headers, timeout=5, allow_redirects=True)
        if resp.status_code != 200:
            return None, None

        resp.encoding = 'utf-8'
        soup = BeautifulSoup(resp.text, 'html.parser')

        # 1. 提取文本
        texts = []
        summary_div = soup.find('div', class_='lemma-summary')
        if summary_div:
            texts.append(summary_div.get_text().strip())

        basic_info = soup.find('div', class_='basic-info')
        if basic_info:
            names = basic_info.find_all('dt')
            values = basic_info.find_all('dd')
            for n, v in zip(names, values):
                texts.append(f"{n.get_text().strip()}: {v.get_text().strip()}")

        summary_text = "\n".join(texts)
        if not summary_text: return None, None

        # 2. 尝试从百科提取图片
        image_url = None
        meta_img = soup.find('meta', property="og:image")
        if meta_img:
            image_url = meta_img.get("content")

        if not image_url:
            pic_div = soup.find('div', class_='summary-pic')
            if pic_div:
                img = pic_div.find('img')
                if img: image_url = img.get('src')

        if image_url:
            if image_url.startswith('//'):
                image_url = "https:" + image_url
            elif image_url.startswith('/'):
                image_url = "https://baike.baidu.com" + image_url

        return summary_text, image_url

    except Exception as e:
        print(f"⚠️ Baidu Baike fetch error: {e}")
        return None, None


def _wiki_search(q: str, lang="en") -> Optional[str]:
    try:
        params = {"action": "opensearch", "search": q, "limit": 1, "namespace": 0, "format": "json"}
        r = requests.get(WIKI_SEARCH.format(lang=lang), params=params, timeout=5)
        if r.status_code == 200:
            j = r.json()
            if isinstance(j, list) and len(j) >= 2 and j[1]: return j[1][0]
    except Exception:
        pass
    return None


def _wiki_summary(title: str, lang="en") -> Optional[Dict[str, Any]]:
    try:
        url = WIKI_SUMMARY.format(lang=lang, title=title.replace(" ", "_"))
        r = requests.get(url, timeout=5, headers={"accept": "application/json"})
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def _expand_queries(user_text: str) -> List[str]:
    qs: List[str] = [user_text.strip()]
    for seg in re.findall(r"[一-龥A-Za-z0-9·\-\s]{2,}", user_text):
        s = seg.strip()
        if s and s not in qs: qs.append(s)
    return list(dict.fromkeys(qs))


def _langs_for(q: str, user_text: str) -> List[str]:
    has_chinese = any('\u4e00' <= ch <= '\u9fff' for ch in user_text + q)
    return ["zh", "en"] if has_chinese else ["en", "zh"]