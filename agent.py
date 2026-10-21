# -*- coding: utf-8 -*-
"""校园二手书智能助手 —— 核心逻辑。

职责：
1. 调用 DeepSeek（文本 + 视觉）
2. 查询 / 检索本地书库
3. 轻量 RAG 检索 FAQ 并生成回答
4. 组合出：拍照识书、找书、估价、文案、教材核验、议价、安全提醒、客服
"""
import base64
import json
import os
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"


# ---------------- 数据加载 ----------------
def _load_json(name):
    # 兼容两种上传结构：标准 data/ 子目录，或被 GitHub 网页拖拽上传拍平到根目录的情况
    for base in (DATA_DIR, BASE_DIR):
        p = base / name
        if p.exists():
            with open(p, encoding="utf-8") as f:
                return json.load(f)
    raise FileNotFoundError(f"找不到数据文件 {name}，请确认它放在项目 data/ 目录下")


BOOKS = _load_json("books.json")
FAQ = _load_json("faq.json")


# ---------------- 配置读取（本地 .env / 云端 st.secrets） ----------------
def _cfg(key, default=""):
    """优先读云端密钥（Streamlit Cloud 的 st.secrets），否则回退到本地 .env / 环境变量。"""
    try:
        import streamlit as st
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key, default)


# ---------------- 模型客户端 ----------------
def has_api_key() -> bool:
    return bool(_cfg("DEEPSEEK_API_KEY"))


def model_name() -> str:
    return _cfg("DEEPSEEK_MODEL", "deepseek-chat")


def vision_model_name() -> str:
    return _cfg("DEEPSEEK_VISION_MODEL", "deepseek-flash")


def base_url() -> str:
    return _cfg("DEEPSEEK_BASE_URL", "https://api.deepseek.com")


def _client():
    return OpenAI(api_key=_cfg("DEEPSEEK_API_KEY"), base_url=base_url())


def call_llm(messages, temperature=0.7, max_tokens=2000) -> str:
    resp = _client().chat.completions.create(
        model=model_name(),
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content


def call_vision(image_bytes, prompt, media_type="image/jpeg", temperature=0.2, max_tokens=1000) -> str:
    """把图片 base64 后发给视觉模型，返回模型的文字描述。"""
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    resp = _client().chat.completions.create(
        model=vision_model_name(),
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{b64}"}},
            ],
        }],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content


# ---------------- 防幻觉系统提示词（第四阶段优化） ----------------
SYSTEM_PROMPT = (
    "你是校园二手书交易智能体。务必遵守以下原则：\n"
    "1. 只基于给定信息回答；信息里没有的事实（书名、出版社、年份、ISBN、价格、成交数据、文献）一律不要编造，"
    "拿不准就直说「这个我不确定」。\n"
    "2. 涉及具体数字（价格、折数、估价）时，优先使用本地书库给定数据，不要凭空估算权威行情。\n"
    "3. 回答简洁、口语化，像学长学姐给建议，避免车轱辘话。"
)


def _grounded_llm(prompt, temperature=0.5, max_tokens=2000) -> str:
    """带防幻觉系统提示词地调用模型（所有涉及事实/价格的回答都走这里）。"""
    return call_llm([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ], temperature=temperature, max_tokens=max_tokens)


# ---------------- 通用小工具 ----------------
def _ngrams(text, n=2):
    text = (text or "").lower()
    return set(text[i:i + n] for i in range(len(text) - n + 1))


def _parse_json(text):
    """尽量从模型输出里提取出一个 JSON 对象。"""
    t = (text or "").strip()
    t = t.replace("```json", "").replace("```", "")
    start, end = t.find("{"), t.rfind("}")
    if start != -1 and end != -1:
        try:
            return json.loads(t[start:end + 1])
        except json.JSONDecodeError:
            return {}
    return {}


# ---------------- 书库查询 / 检索 ----------------
def lookup_book(query):
    """按 ISBN 或书名精确/包含匹配，找不到返回 None。"""
    q = query.strip()
    q_norm = q.replace("-", "").replace(" ", "").lower()
    for b in BOOKS:
        if b["isbn"].replace("-", "").lower() == q_norm:
            return b
    for b in BOOKS:
        if b["title"].lower() in q.lower() or q.lower() in b["title"].lower():
            return b
    return None


def search_books(query, k=10):
    """多字段打分检索，返回按相关度排序的书列表（用于“找书”）。"""
    q = (query or "").strip().lower()
    if not q:
        return BOOKS[:k]
    q_norm = q.replace("-", "").replace(" ", "")
    scored = []
    for b in BOOKS:
        score = 0
        if q_norm and b["isbn"].replace("-", "").lower() == q_norm:
            score += 100
        if q in b["title"].lower():
            score += 50
        elif b["title"].lower() in q:
            score += 40
        if q in b["author"].lower():
            score += 30
        if q in (b.get("course") or "").lower():
            score += 20
        if q in (b.get("major") or "").lower():
            score += 15
        hay = " ".join([b["title"], b["author"], b.get("course", ""), b.get("major", "")]).lower()
        score += len(_ngrams(q) & _ngrams(hay))
        if score > 0:
            scored.append((score, b))
    scored.sort(key=lambda x: -x[0])
    return [b for _, b in scored[:k]]


def list_courses():
    seen, out = set(), []
    for b in BOOKS:
        c = b.get("course") or "通用"
        if c not in seen:
            seen.add(c)
            out.append(c)
    return sorted(out)


def list_majors():
    seen, out = set(), []
    for b in BOOKS:
        m = b.get("major") or "其他"
        if m not in seen:
            seen.add(m)
            out.append(m)
    return sorted(out)


def books_by_major(major):
    return [b for b in BOOKS if (b.get("major") or "其他") == major]


# ---------------- 拍照识书 ----------------
def recognize_book(image_bytes, media_type="image/jpeg"):
    """识别书籍照片，返回 (匹配到的书 or None, 识别出的信息 dict)。"""
    prompt = (
        "这是一本书籍的照片。请识别书籍信息，只返回一个 JSON 对象，格式如下，"
        "不要输出任何其他文字或代码块：\n"
        '{"title":"书名","author":"作者","publisher":"出版社","edition":"版次","isbn":"封底ISBN，若看不清填空字符串"}\n'
        "如果某项看不清，就填空字符串。"
    )
    raw = call_vision(image_bytes, prompt, media_type=media_type)
    info = _parse_json(raw)
    return _match_book(info), info


def _match_book(info):
    title = (info.get("title") or "").strip()
    isbn = (info.get("isbn") or "").replace("-", "").replace(" ", "").lower()
    if isbn:
        for b in BOOKS:
            if b["isbn"].replace("-", "").lower() == isbn:
                return b
    if title:
        for b in BOOKS:
            if title.lower() in b["title"].lower() or b["title"].lower() in title.lower():
                return b
    return None


# ---------------- 估价 ----------------
_COND_FACTOR = {"全新": 1.0, "九五新": 0.9, "八五新": 0.75, "七成新": 0.6}


def estimate_price(book, condition="八五新"):
    factor = _COND_FACTOR.get(condition, 0.75)
    low = round(book["secondhand_price_low"] * factor)
    high = round(book["secondhand_price_high"] * factor)
    prompt = (
        f"你是校园二手书交易智能体，请用 2-3 句口语化的话给出估价说明（像学长学姐给建议）。\n"
        f"书名：《{book['title']}》\n"
        f"出版社：{book['publisher']}  版次：{book['edition']}\n"
        f"原价：{book['original_price']} 元  品相：{condition}\n"
        f"参考二手价区间：{low}-{high} 元\n"
        f"要求：说明为什么是这个价（基于原价、品相、版次折旧），可提示折价规律；"
        f"严禁编造任何成交数据、行情数字或未给出的出版信息，拿不准就直说。"
    )
    explanation = _grounded_llm(prompt)
    return low, high, explanation


def estimate_price_unknown(info, condition="八五新"):
    """书库里没匹配到时，根据识别到的信息让模型给一个粗略估价。"""
    prompt = (
        f"你是校园二手书交易智能体。有一本暂时不在书库里的书，请根据识别信息给出一个合理的二手估价区间（元）和一句说明。\n"
        f"识别信息：书名《{info.get('title', '未知')}》 作者 {info.get('author', '未知')} "
        f"出版社 {info.get('publisher', '未知')} 版次 {info.get('edition', '未知')}\n"
        f"品相：{condition}\n"
        f"要求：输出格式类似「估价区间：XX-XX 元。说明：……」；"
        f"识别信息不全时不要硬猜出版社、年份、ISBN，说明按通用教材折价规律估算即可。"
    )
    return _grounded_llm(prompt, temperature=0.5)


# ---------------- 文案生成 ----------------
def generate_listing(book, condition, notes=""):
    prompt = (
        f"你是校园二手书交易智能体，请为这本书生成一条可直接发到校园二手群的出售文案。\n"
        f"书名：《{book['title']}》  作者：{book['author']}\n"
        f"出版社：{book['publisher']}  版次：{book['edition']}\n"
        f"原价：{book['original_price']} 元  适用课程：{book.get('course', '通用')}\n"
        f"品相：{condition}  补充说明：{notes or '无'}\n\n"
        f"要求：\n"
        f"1. 标题抓眼球（含书名和版次）；\n"
        f"2. 正文 80-150 字，口语化；\n"
        f"3. 结构 = 书籍亮点 + 品相说明 + 适用课程 + 价格提示，但句式不要照搬固定模板；\n"
        f"4. 避免「九五新/无笔记/大一新生/考研党」这类套话堆砌，换具体、真实一点的表达；\n"
        f"5. 严禁编造不存在的赠品、服务、成交记录或库存信息。"
    )
    return _grounded_llm(prompt, temperature=0.8)


# ---------------- FAQ 检索（轻量 RAG）----------------
def retrieve_faq(query, k=3):
    q_tokens = _ngrams(query)
    scored = []
    for item in FAQ:
        hay = _ngrams(item["q"]) | _ngrams(" ".join(item.get("keywords", [])))
        scored.append((len(q_tokens & hay), item))
    scored.sort(key=lambda x: -x[0])
    hits = [item for score, item in scored[:k] if score > 0]
    return hits or FAQ[:2]


def answer_faq(question):
    context = "\n".join(f"Q: {h['q']}\nA: {h['a']}" for h in retrieve_faq(question))
    prompt = (
        f"你是校园二手书交易智能体的客服助手，请参考下面检索到的资料回答用户问题。\n\n"
        f"【资料】\n{context}\n\n"
        f"【用户问题】\n{question}\n\n"
        f"要求：简洁、准确、口语化；只依据【资料】回答，资料没有的内容直接说「这个我不清楚」，"
        f"绝不编造规则或价格，并建议人工核实。"
    )
    return _grounded_llm(prompt, temperature=0.5)


# ---------------- 教材核验（多本推荐）----------------
def verify_version(course):
    c = course.strip()
    matched = []
    seen = set()
    for b in BOOKS:
        bc = b.get("course") or ""
        if c and (c in bc or bc in c):
            if b["isbn"] not in seen:
                seen.add(b["isbn"])
                matched.append(b)
    if not matched:
        return [], None
    lines = "\n".join(
        f"- 《{b['title']}》{b['edition']} · {b['publisher']} · ISBN {b['isbn']}"
        for b in matched
    )
    prompt = (
        f"课程《{c}》的常用教材有以下几本，请用 2-3 句口语化的话提醒学生："
        f"购买前务必以老师给出的 ISBN 和版次为准，避免买错版本。\n{lines}\n"
        f"要求：只基于上面列出的教材做提醒，不要补充列表之外的教材或版次信息。"
    )
    note = _grounded_llm(prompt, temperature=0.5)
    return matched, note


# ---------------- 议价助手 ----------------
def negotiate(book, seller_price, condition="八五新"):
    prompt = (
        f"你是校园二手书交易智能体，帮买家分析这笔二手书交易是否划算并给出议价建议。\n"
        f"书名：《{book['title']}》{book['edition']} · {book['publisher']}\n"
        f"原价：{book['original_price']} 元  卖家报价：{seller_price} 元  品相：{condition}\n"
        f"本地参考二手价区间：{book['secondhand_price_low']}-{book['secondhand_price_high']} 元\n\n"
        f"要求：1. 基于「参考区间 × 品相」先算出建议目标价并说明依据；2. 判断报价偏高/合理/偏低；"
        f"3. 给 1-2 句议价话术，口语化，控制在 4 句以内；"
        f"4. 严禁用「谎称别人卖得更低」这类虚假锚点，也不编造行情数据。"
    )
    return _grounded_llm(prompt, temperature=0.7)


# ---------------- 交易安全 ----------------
_RISK_RULES = [
    ("私下转账/先款", ["私下转账", "先款", "直接转账", "微信转", "支付宝转", "先付款", "先打钱", "先转"]),
    ("超低价", ["超低价", "白菜价", "骨折价", "免费送", "低到离谱", "太便宜"]),
    ("陌生链接/二维码", ["链接", "点开", "扫码", "网址", "二维码", "下载app", "下载APP"]),
    ("催单施压", ["赶紧", "马上", "手慢无", "最后一件", "别人要了", "不买没了"]),
    ("先确认收货", ["先确认收货", "先好评", "先确认", "确认收货"]),
]


def safety_check(text):
    t = text or ""
    hits = [name for name, kws in _RISK_RULES if any(k in t for k in kws)]
    if hits:
        warn = "；".join(hits)
        prompt = (
            f"你是校园二手书交易安全助手。买家描述的场景里检测到风险点：{warn}。\n"
            f"场景：{t}\n请用 2-4 句口语化的话给出具体避坑提醒（坚持平台担保、不私下转账、当面验货、保留记录等）。"
        )
        advice = _grounded_llm(prompt, temperature=0.5)
        return hits, advice
    prompt = (
        f"你是校园二手书交易安全助手。买家描述的交易场景：{t}\n"
        f"请用 2-3 句口语化的话给出通用安全提醒（平台担保、当面验货、保留聊天记录）。"
    )
    advice = _grounded_llm(prompt, temperature=0.5)
    return [], advice
