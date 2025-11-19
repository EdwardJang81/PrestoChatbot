import streamlit as st
from google import genai
from google.genai import types, errors
import time
import os

# ==== 설정 부분 ====
api_key = os.getenv("GOOGLE_API_KEY")
STORE_DISPLAY_NAME = "presto_docs_store"

if not api_key:
    st.error("환경 변수 GOOGLE_API_KEY가 설정되어 있지 않습니다. (GOOGLE_API_KEY)")
    st.stop()

# ==== 클라이언트 & 스토어 ====
@st.cache_resource
def get_client():
    return genai.Client(api_key=api_key)

@st.cache_resource
def get_store(display_name: str):
    client = get_client()
    for s in client.file_search_stores.list():
        if getattr(s, "display_name", None) == display_name:
            return s
    # 없으면 생성
    store = client.file_search_stores.create(
        config={"display_name": display_name}
    )
    return store


def ask_question(store_name: str, history: list[types.Content]) -> str:
    """File Search가 연결된 Gemini에게 질문 (재시도 로직 포함)"""
    client = get_client()
    max_retries = 5
    delay = 2  # 초

    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model="gemini-3-pro-preview", #"gemini-2.5-flash",
                contents=history,
                config=types.GenerateContentConfig(
                    tools=[
                        types.Tool(
                            file_search=types.FileSearch(
                                file_search_store_names=[store_name]
                            )
                        )
                    ]
                ),
            )
            return response.text or ""

        except errors.ServerError as e:
            # 503 같은 서버 과부하 에러만 재시도
            if "overloaded" in str(e) and attempt < max_retries:
                print(
                    f"[ServerError] 과부하, {attempt}회 시도 실패 → {delay}초 후 재시도"
                )
                time.sleep(delay)
                continue

            st.error("현재 Gemini 서버가 과부하 상태입니다. 잠시 후 다시 시도해 주세요.")
            print("[ServerError 최종 실패]", e)
            return ""

        except errors.APIError as e:
            st.error(f"Gemini API 에러 발생: {e}")
            print("[APIError]", e)
            return ""


# ==== Streamlit UI ====
st.set_page_config(page_title="Presto Docs Chat", page_icon="🤖", layout="wide")
st.title("📘 Presto Docs Chat (File Search Preview)")

store = get_store(STORE_DISPLAY_NAME)

# --- 세션 상태 초기화 ---
if "history" not in st.session_state:
    st.session_state.history = []  # type: list[types.Content]

if "last_request_time" not in st.session_state:
    st.session_state.last_request_time = 0.0

# --- 사이드바: 질문 히스토리 (ChatGPT 왼쪽 리스트 느낌) ---
st.sidebar.header("질문 히스토리")
user_messages = [m.parts[0].text for m in st.session_state.history if m.role == "user"]

if user_messages:
    for i, q in enumerate(user_messages[-20:], 1):
        # 마지막 20개만 표시
        st.sidebar.markdown(f"**{i}.** {q}")
else:
    st.sidebar.caption("아직 질문이 없습니다.")

# --- 메인 영역: 기존 대화 표시 (ChatGPT 대화창 느낌) ---
for msg in st.session_state.history:
    if msg.role == "user":
        with st.chat_message("user"):
            st.markdown(msg.parts[0].text)
    else:
        with st.chat_message("assistant"):
            st.markdown(msg.parts[0].text)

# --- 입력창 ---
user_input = st.chat_input("질문을 입력하세요. (예: ACSPL에서 Enable 사용하는 법 알려줘)")

MAX_TURNS = 6  # user+assistant 쌍 6개 정도면 충분

if user_input:
    now = time.time()
    # 요청 간격 제한 (너무 빠른 연타 방지)
    if now - st.session_state.last_request_time < 1.5:
        st.warning("요청 간격이 너무 짧습니다. 잠시 후 다시 입력해 주세요.")
    else:
        st.session_state.last_request_time = now

        # 1) 화면에 유저 메시지 표시 + history 반영
        user_msg = types.Content(
            role="user",
            parts=[types.Part(text=user_input)]
        )
        st.session_state.history.append(user_msg)

        with st.chat_message("user"):
            st.markdown(user_input)

        # 2) 모델 호출
        with st.chat_message("assistant"):
            with st.spinner("생각 중..."):
                answer = ask_question(store.name, st.session_state.history)
                if answer:
                    st.markdown(answer)
                else:
                    st.markdown("⚠️ 응답을 가져오지 못했습니다. (서버 과부하 또는 API 에러)")

        # 3) 모델 응답도 history에 추가
        model_msg = types.Content(
            role="model",
            parts=[types.Part(text=answer)]
        )
        st.session_state.history.append(model_msg)

        # 히스토리가 너무 길어지면 뒤에서 N턴만 남기기
        if len(st.session_state.history) > MAX_TURNS * 2:
            st.session_state.history = st.session_state.history[-MAX_TURNS * 2:]
