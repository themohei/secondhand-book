# -*- coding: utf-8 -*-
"""校园二手书助手 —— Streamlit 界面。"""
import io

import streamlit as st
from PIL import Image

import agent

st.set_page_config(page_title="校园二手书助手", page_icon="📚", layout="wide")

st.title("📚 校园二手书助手")
st.caption("买卖二手教材，一个就够")


def require_key() -> bool:
    if not agent.has_api_key():
        st.error("⚠️ 智能服务暂未连接，请稍后再试。")
        return False
    return True


def prepare_image(file_bytes):
    """压缩并统一转成 JPEG，减小传给视觉模型的体积。"""
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    max_side = 1280
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def pick_book(label="书名 / ISBN", key=""):
    """搜索式选书：输入关键词 → 从结果里选一本。"""
    q = st.text_input(label, key=f"{key}_q", placeholder="例如：高等数学 或 9787302023685")
    if q.strip():
        results = agent.search_books(q, k=8)
        if results:
            titles = [f"《{b['title']}》{b['edition']}" for b in results]
            choice = st.radio("选择书籍", titles, key=f"{key}_r", index=0)
            idx = titles.index(choice) if choice in titles else 0
            return results[idx]
    return None


CONDITIONS = ["全新", "九五新", "八五新", "七成新"]


tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    ["📸 拍照估价", "🔍 找书", "🎓 教材核验", "🤝 议价助手", "🛡️ 交易安全", "✍️ 发布文案", "💬 智能客服"]
)

# ---------- Tab1 拍照估价 ----------
with tab1:
    st.subheader("拍一张书，秒出估价")
    col_a, col_b = st.columns([1, 1])
    with col_a:
        uploaded = st.file_uploader("上传书籍封面或封底照片", type=["jpg", "jpeg", "png", "webp", "gif"])
        if uploaded is not None:
            st.image(uploaded, use_container_width=True)
    with col_b:
        condition = st.selectbox("品相", CONDITIONS, index=2)
        if st.button("拍照识别并估价", type="primary", disabled=uploaded is None):
            if require_key():
                img_bytes = prepare_image(uploaded.getvalue())
                with st.spinner("正在识别书籍…"):
                    book, info = agent.recognize_book(img_bytes)
                if book:
                    st.success(f"识别到：《{book['title']}》{book['edition']} · {book['publisher']}")
                    with st.spinner("正在估价…"):
                        low, high, exp = agent.estimate_price(book, condition)
                    st.metric("参考估价", f"¥{low} ~ ¥{high}", delta=f"原价 ¥{book['original_price']}")
                    st.info(exp)
                else:
                    st.warning("识别到了信息，但书库里暂时没有这本书：")
                    st.write(info)
                    if info.get("title"):
                        with st.spinner("正在估算…"):
                            rough = agent.estimate_price_unknown(info, condition)
                        st.info(rough)
        st.divider()
        st.markdown("**或者直接输入 ISBN / 书名：**")
        query = st.text_input("ISBN 或书名", placeholder="例如：9787302023685 或 数据结构")
        if st.button("开始估价"):
            if not query.strip():
                st.warning("请先输入 ISBN 或书名")
            elif require_key():
                book = agent.lookup_book(query)
                if book is None:
                    st.warning("书库里没找到这本书，换个关键词试试（如：高等数学）。")
                else:
                    st.success(f"已找到：《{book['title']}》{book['edition']} · {book['publisher']}")
                    with st.spinner("正在估价…"):
                        low, high, exp = agent.estimate_price(book, condition)
                    st.metric("参考估价", f"¥{low} ~ ¥{high}", delta=f"原价 ¥{book['original_price']}")
                    st.info(exp)

# ---------- Tab2 找书 ----------
with tab2:
    st.subheader("找书：搜索或按专业浏览")
    q = st.text_input("输入书名 / 作者 / ISBN / 课程", placeholder="例如：数据结构 或 谢希仁 或 高等数学")
    if q.strip():
        results = agent.search_books(q, k=12)
        if not results:
            st.info("没找到相关书籍，换个关键词试试。")
        for b in results:
            with st.container(border=True):
                st.markdown(f"**《{b['title']}》{b['edition']}**")
                st.caption(f"{b['author']} · {b['publisher']} · 适用课程：{b.get('course', '通用')}")
                st.markdown(
                    f"原价 ¥{b['original_price']} ｜ 二手参考 ¥{b['secondhand_price_low']} ~ ¥{b['secondhand_price_high']} ｜ ISBN `{b['isbn']}`"
                )
    else:
        majors = agent.list_majors()
        if majors:
            st.markdown("**或按专业浏览：**")
            major = st.selectbox("选择专业", majors)
            for b in agent.books_by_major(major):
                with st.container(border=True):
                    st.markdown(f"**《{b['title']}》{b['edition']}**")
                    st.caption(f"{b['author']} · {b['publisher']} · 适用课程：{b.get('course', '通用')}")
                    st.markdown(
                        f"原价 ¥{b['original_price']} ｜ 二手参考 ¥{b['secondhand_price_low']} ~ ¥{b['secondhand_price_high']} ｜ ISBN `{b['isbn']}`"
                    )

# ---------- Tab3 教材核验 ----------
with tab3:
    st.subheader("教材核验：这门课用哪几本？")
    course = st.text_input("输入课程名", placeholder="例如：高等数学 或 数据结构")
    st.caption("常见课程：" + "、".join(agent.list_courses()[:12]) + " …")
    if st.button("查询教材", type="primary"):
        if not course.strip():
            st.warning("请输入课程名")
        elif require_key():
            books, note = agent.verify_version(course)
            if not books:
                st.warning("暂未收录该课程，试试：高等数学 / 数据结构 / 计算机网络 / 大学物理 / 微观经济学")
            else:
                st.success(f"「{course}」常用教材 {len(books)} 本")
                for b in books:
                    with st.container(border=True):
                        st.markdown(f"**《{b['title']}》{b['edition']}**")
                        st.caption(f"{b['author']} · {b['publisher']}")
                        st.markdown(
                            f"ISBN `{b['isbn']}` ｜ 二手参考 ¥{b['secondhand_price_low']} ~ ¥{b['secondhand_price_high']}"
                        )
                if note:
                    st.info(note)

# ---------- Tab4 议价助手 ----------
with tab4:
    st.subheader("议价助手：这笔买卖划算吗？")
    book = pick_book("书名 / ISBN", key="neg")
    if book:
        st.caption(f"已选：《{book['title']}》{book['edition']}，原价 ¥{book['original_price']}，二手参考 ¥{book['secondhand_price_low']} ~ ¥{book['secondhand_price_high']}")
        price = st.number_input("卖家报价（元）", min_value=0, value=int(book["secondhand_price_high"]))
        cond = st.selectbox("品相", CONDITIONS, index=2, key="neg_cond")
        if st.button("帮我分析", type="primary"):
            if require_key():
                with st.spinner("正在分析…"):
                    advice = agent.negotiate(book, price, cond)
                st.success("议价建议")
                st.info(advice)

# ---------- Tab5 交易安全 ----------
with tab5:
    st.subheader("交易安全：一眼识破套路")
    scene = st.text_area(
        "描述你要进行的交易",
        placeholder="例如：对方让我微信先转账 200 元，说收到就发货，还发了个链接让我点。",
    )
    if st.button("检测风险", type="primary"):
        if not scene.strip():
            st.warning("请先描述交易场景")
        elif require_key():
            with st.spinner("正在分析…"):
                hits, advice = agent.safety_check(scene)
            if hits:
                st.error("⚠️ 检测到风险点：" + "、".join(hits))
            else:
                st.success("未发现明显风险点")
            st.info(advice)

# ---------- Tab6 发布文案 ----------
with tab6:
    st.subheader("发布文案：一键写好商品描述")
    book = pick_book("书名 / ISBN", key="listing")
    cond = st.selectbox("品相", CONDITIONS, index=2, key="list_cond")
    notes = st.text_input("补充说明（可选）", placeholder="例如：有少量铅笔笔记，书脊完好")
    if book is None:
        st.caption("先输入书名查找，再生成文案。")
    elif st.button("生成文案", type="primary"):
        if require_key():
            with st.spinner("正在撰写…"):
                listing = agent.generate_listing(book, cond, notes)
            st.success("文案已生成，复制到二手群即可：")
            st.code(listing, language=None)

# ---------- Tab7 智能客服 ----------
with tab7:
    st.subheader("智能客服")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
    q = st.chat_input("问我关于二手书的问题，例如：八五新是什么意思？")
    if q:
        st.session_state.messages.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            if not require_key():
                ans = "（服务暂未连接，请稍后再试）"
            else:
                with st.spinner("思考中…"):
                    ans = agent.answer_faq(q)
            st.markdown(ans)
        st.session_state.messages.append({"role": "assistant", "content": ans})
