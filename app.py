import streamlit as st
import re
import numpy as np
from collections import deque
from typing import List, Dict, Tuple
from pinecone import Pinecone
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# =====================================
# 🔐 Secrets 설정 (Streamlit Cloud)
# =====================================
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_INDEX_NAME = st.secrets["PINECONE_INDEX_NAME"]
EMBEDDING_MODEL_NAME = st.secrets["EMBEDDING_MODEL_NAME"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
LLM_MODEL_NAME = st.secrets["LLM_MODEL_NAME"]

DEFAULT_VECTOR_WEIGHT = float(st.secrets.get("DEFAULT_VECTOR_WEIGHT", 0.7))
DEFAULT_BM25_WEIGHT = float(st.secrets.get("DEFAULT_BM25_WEIGHT", 0.3))
DEFAULT_TOP_K = int(st.secrets.get("DEFAULT_TOP_K", 50))
DEFAULT_CONTEXT_CHARS = int(st.secrets.get("DEFAULT_CONTEXT_CHARS", 3000))
DEFAULT_CONTEXT_TOP_N = int(st.secrets.get("DEFAULT_CONTEXT_TOP_N", 5))

# =====================================
# ⚙️ Streamlit 기본 설정
# =====================================
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")
st.title("💬 RAG Chatbot (Pinecone + OpenAI)")
st.caption("Sentence-Transformer + BM25 + OpenAI 기반의 RAG 챗봇")

# =====================================
# 🔗 Pinecone 연결
# =====================================
@st.cache_resource
def load_pinecone_index():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    return pc.Index(PINECONE_INDEX_NAME)

try:
    index = load_pinecone_index()
    st.success(f"Pinecone Index 연결됨 ✅ ({PINECONE_INDEX_NAME})")
except Exception as e:
    st.error(f"Pinecone 연결 실패 ❌: {e}")
    st.stop()

# =====================================
# 🔤 SentenceTransformer 임베딩 모델
# =====================================
@st.cache_resource
def load_embedder():
    return SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")

model = load_embedder()

# =====================================
# 🔍 검색 함수
# =====================================
def simple_tokenize(s: str):
    return re.findall(r"[A-Za-z0-9가-힣]+", (s or "").lower())

def vector_search(query: str, top_k: int = 50, meta_filter=None):
    q_vec = model.encode([f"query: {query}"], convert_to_numpy=True, normalize_embeddings=True)[0]
    kwargs = {
        "vector": q_vec.tolist(),
        "top_k": top_k,
        "include_values": False,
        "include_metadata": True,
    }
    if meta_filter:
        kwargs["filter"] = meta_filter
    res = index.query(**kwargs)
    return [(m["id"], float(m["score"]), m.get("metadata", {})) for m in res.get("matches", [])]

def bm25_over_candidates(query: str, candidates):
    ids, docs = [], []
    for cid, _, meta in candidates:
        text = (meta or {}).get("text_content") or ""
        if not text:
            title = (meta or {}).get("title") or ""
            keywords = (meta or {}).get("keywords") or ""
            text = f"{title}\n{keywords}"
        ids.append(cid)
        docs.append(simple_tokenize(text))
    if not docs:
        return {}
    bm25 = BM25Okapi(docs)
    scores = bm25.get_scores(simple_tokenize(query)) if query else np.zeros(len(ids))
    max_b = float(np.max(scores)) if len(scores) else 0.0
    return {ids[i]: (float(scores[i]) / max_b if max_b > 0 else 0.0) for i in range(len(ids))}

# =====================================
# 🧠 RAG + LLM 로직
# =====================================
openai_client = OpenAI(api_key=OPENAI_API_KEY)

SYSTEM_PROMPT = (
    "너는 RAG 기반 도우미야. 제공된 컨텍스트를 우선 활용해서 간결하고 정확하게 답해.\n"
    "근거가 없으면 솔직히 모른다고 말해.\n"
    "답변 끝에는 출처를 bullet 형식으로 표시해."
)

def build_context_improved(query, candidates, vec_w, bm25_w, top_n, max_chars):
    bm25_scores = bm25_over_candidates(query, candidates)
    scored = []
    for cid, v_score, meta in candidates:
        b_score = bm25_scores.get(cid, 0.0)
        combo = vec_w * float(v_score) + bm25_w * float(b_score)
        scored.append((combo, cid, meta))
    scored.sort(reverse=True, key=lambda x: x[0])

    picked, used = [], 0
    for _, cid, meta in scored[: max(1, int(top_n) * 3)]:
        text = (meta or {}).get("text_content") or (meta or {}).get("title") or ""
        if not text:
            continue
        if used + len(text) > max_chars:
            continue
        picked.append({
            "id": cid,
            "title": (meta or {}).get("title"),
            "source": (meta or {}).get("source_doc"),
            "chunk": text,
        })
        used += len(text)
        if len(picked) >= int(top_n):
            break
    return picked

def call_openai_improved(api_key: str, model: str, messages: List[Dict]) -> str:
    client = OpenAI(api_key=api_key)
    r = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
    )
    return r.choices[0].message.content or ""

# =====================================
# 💬 대화 히스토리
# =====================================
if "history" not in st.session_state:
    st.session_state.history = deque(maxlen=3)

def chat_once(question: str):
    candidates = vector_search(question, top_k=DEFAULT_TOP_K)
    contexts = build_context_improved(
        question,
        candidates,
        DEFAULT_VECTOR_WEIGHT,
        DEFAULT_BM25_WEIGHT,
        DEFAULT_CONTEXT_TOP_N,
        DEFAULT_CONTEXT_CHARS,
    )
    context_text = "\n\n".join([f"[#{i+1}] {c['chunk']}" for i, c in enumerate(contexts)])
    citations = "\n".join(
        [f"- [#{i+1}] {c.get('title') or c.get('source') or c['id']}" for i, c in enumerate(contexts)]
    )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for past_q, past_a, _ in st.session_state.history:
        if past_q:
            messages.append({"role": "user", "content": past_q})
        if past_a:
            messages.append({"role": "assistant", "content": past_a})

    user_msg = (
        f"질문: {question}\n\n컨텍스트:\n{context_text}\n\n"
        "컨텍스트를 기반으로 답하세요. 답 끝에 '출처' 섹션을 넣어 아래 목록에서 근거를 인용하세요.\n"
        f"출처 목록:\n{citations}"
    )
    messages.append({"role": "user", "content": user_msg})

    answer = call_openai_improved(OPENAI_API_KEY, LLM_MODEL_NAME, messages)
    st.session_state.history.append((question, answer, ""))

    return answer.strip(), contexts

# =====================================
# 🧭 Streamlit UI
# =====================================
st.subheader("💭 대화하기")

query = st.text_area("질문을 입력하세요:", height=100, placeholder="예: Pinecone 인덱스는 어떻게 작동하나요?")

if st.button("질문하기", type="primary"):
    if not query.strip():
        st.warning("질문을 입력해주세요.")
    else:
        with st.spinner("답변 생성 중... ⏳"):
            try:
                answer, ctx = chat_once(query)
                st.markdown("### 💬 답변")
                st.write(answer)

                with st.expander("🔍 사용된 컨텍스트 보기"):
                    for i, c in enumerate(ctx, 1):
                        st.markdown(f"**#{i}. {c['title'] or '제목 없음'}**")
                        st.text(c['chunk'][:400] + ("..." if len(c['chunk']) > 400 else ""))
            except Exception as e:
                st.error(f"오류 발생: {e}")

# =====================================
# ⚙️ 사이드바
# =====================================
with st.sidebar:
    st.header("⚙️ 설정")
    st.info("Secrets.toml에 API 키를 저장하면 자동으로 불러옵니다.")
    st.write("현재 Pinecone 인덱스:", PINECONE_INDEX_NAME)
    if st.button("💣 대화 초기화"):
        st.session_state.history.clear()
        st.success("대화 기록이 초기화되었습니다.")
