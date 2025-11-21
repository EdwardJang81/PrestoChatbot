import streamlit as st
from google import genai
from google.genai import types, errors
import time
import os

# ==== 설정 부분 ====
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("환경 변수 GOOGLE_API_KEY가 설정되어 있지 않습니다. (GOOGLE_API_KEY)")
    st.stop()

# ==== 클라이언트 & 스토어 ====
@st.cache_resource
def get_client():
    return genai.Client(api_key=api_key)


@st.cache_resource
def get_store(display_name: str):
    """
    display_name으로 Store를 찾고 없으면 새로 생성.
    (이미 만들어둔 presto_* 스토어도 display_name 기준으로 잘 찾아옵니다.)
    """
    client = get_client()
    for s in client.file_search_stores.list():
        if getattr(s, "display_name", None) == display_name:
            return s

    # 없으면 생성 (예외 케이스용)
    store = client.file_search_stores.create(
        config={"display_name": display_name}
    )
    return store


@st.cache_data(show_spinner=False)
def list_documents(store_name: str):
    """
    특정 File Search Store 안에 들어있는 문서 리스트 조회.
    반환값: documents 리스트
    """
    client = get_client()
    docs = list(client.file_search_stores.documents.list(parent=store_name))
    return docs


def ask_question(
    store_name: str,
    history: list[types.Content],
    model_name: str,
) -> str:
    """File Search가 연결된 Gemini에게 질문 (재시도 로직 포함)"""
    client = get_client()
    max_retries = 5
    delay = 2  # 초

    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=history,
                config=types.GenerateContentConfig(
                    system_instruction=(
                        "당신은 제공된 문서를 기반으로 답변하는 AI 어시스턴트입니다.\n"
                        "다음 규칙을 반드시 준수하세요:\n"
                        "1. 오직 제공된 문서(Context)에 있는 내용만 사용하여 답변하세요.\n"
                        "2. 문서에 없는 내용은 '문서에 해당 내용이 없습니다'라고 답변하고, 외부 지식을 사용하지 마세요.\n"
                        "3. 답변의 끝에는 반드시 참고한 문서의 이름(Source)을 명시하세요.\n"
                        "   예시: (출처: 파일명.pdf)\n"
                        "4. 답변은 반드시 한국어로 작성하세요."
                    ),
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
st.set_page_config(page_title="Presto Knowledge AI Copilot", page_icon="🤖", layout="wide")
st.title("📘 Presto Knowledge AI Copilot")

# --- 세션 상태 초기화 ---
if "history" not in st.session_state:
    st.session_state.history = []  # type: list[types.Content]

if "last_request_time" not in st.session_state:
    st.session_state.last_request_time = 0.0

MAX_TURNS = 6  # user+assistant 쌍 6개 정도면 충분

# --- 사이드바: 스토어 / 모델 선택 + 파일 리스트 ---
st.sidebar.header("⚙️ 설정")

# 1) Store 선택 (한글 라벨 ↔ 실제 store display_name 매핑)
store_options = {
    "[기술]제품": "presto_products",
    "[기술]어플리케이션": "presto_applications",
    "[기술]프로그래밍": "presto_programmings",
    #"[회사]사내규정": "presto_regulations",
}

selected_label = st.sidebar.selectbox(
    "📂 Documentation Store 선택",
    options=list(store_options.keys()),
    index=0,
)

store_display_name = store_options[selected_label]
store = get_store(store_display_name)

# 2) 모델 선택
model_name = st.sidebar.selectbox(
    "🧠 Gemini 모델 선택",
    options=[
        "gemini-2.5-flash",        # 기본값 (빠른 응답)
        "gemini-2.5-pro",          # 정확도/추론력 우선
        "gemini-3-pro-preview",    # 최신 미리보기
    ],
    index=0,
)

# 3) 선택된 Store 안의 파일 리스트 표시
st.sidebar.subheader("📄 선택된 Store의 파일 목록")

docs = list_documents(store.name)
if docs:
    for d in docs:
        display = getattr(d, "display_name", None) or getattr(d, "name", "(no name)")
        st.sidebar.markdown(f"- `{display}`")
else:
    st.sidebar.caption("아직 이 Store에는 등록된 파일이 없습니다.")


# --- 메인 영역 컨테이너: 위(대화), 아래(질문 박스) ---
chat_container = st.container()  # 대화 표시용 컨테이너

if "history" not in st.session_state:
    st.session_state.history = []  # type: list[types.Content]

if "last_request_time" not in st.session_state:
    st.session_state.last_request_time = 0.0

MAX_TURNS = 6  # user+assistant 쌍 6개 정도면 충분


def render_history_compact():
    """새 질문이 없을 때: 과거 대화는 접고, 마지막 대화만 펼쳐서 보여준다."""
    history = st.session_state.history
    pairs = len(history) // 2

    if pairs == 0:
        return

    with chat_container:
        if pairs == 1:
            user_msg = history[0]
            model_msg = history[1]
            with st.chat_message("user"):
                st.markdown(user_msg.parts[0].text)
            with st.chat_message("assistant"):
                st.markdown(model_msg.parts[0].text)
        else:
            # 이전 대화들: expander로 접기
            for i in range(pairs - 1):
                user_msg = history[2 * i]
                model_msg = history[2 * i + 1]
                title = user_msg.parts[0].text.strip().replace("\n", " ")
                if len(title) > 30:
                    title = title[:27] + "..."
                with st.expander(f"대화 {i+1}: {title}", expanded=False):
                    with st.chat_message("user"):
                        st.markdown(user_msg.parts[0].text)
                    with st.chat_message("assistant"):
                        st.markdown(model_msg.parts[0].text)

            # 마지막(가장 최근) 대화는 그대로 펼쳐서 보여주기
            last_user = history[-2]
            last_model = history[-1]
            with st.chat_message("user"):
                st.markdown(last_user.parts[0].text)
            with st.chat_message("assistant"):
                st.markdown(last_model.parts[0].text)



st.markdown("---")

user_input = st.chat_input(
    placeholder="💬 질문을 입력해 주세요 (Enter로 질문 보내기)"
)

if user_input:
    now = time.time()

    # 요청 간격 제한
    if now - st.session_state.last_request_time < 1.5:
        st.session_state.last_request_time = now
        # 기존 대화는 접어서 보여주고 경고
        render_history_compact()
        with chat_container:
            st.warning("요청 간격이 너무 짧습니다. 잠시 후 다시 입력해 주세요.")
    else:
        st.session_state.last_request_time = now

        # 새 user 메시지 구성
        user_msg = types.Content(
            role="user",
            parts=[types.Part(text=user_input)]
        )
        temp_history = st.session_state.history + [user_msg]

        # 채팅 영역: 과거 대화는 expander, 새 질문/답변은 펼쳐서
        with chat_container:
            history = st.session_state.history
            pairs = len(history) // 2

            # 이전 대화 expander로 접기
            for i in range(pairs):
                prev_user = history[2 * i]
                prev_model = history[2 * i + 1]
                title = prev_user.parts[0].text.strip().replace("\n", " ")
                if len(title) > 30:
                    title = title[:27] + "..."
                with st.expander(f"대화 {i+1}: {title}", expanded=False):
                    with st.chat_message("user"):
                        st.markdown(prev_user.parts[0].text)
                    with st.chat_message("assistant"):
                        st.markdown(prev_model.parts[0].text)

            # 이번에 보낸 user 메시지
            with st.chat_message("user"):
                st.markdown(user_input)

            # assistant 답변 + spinner
            with st.chat_message("assistant"):
                with st.spinner("생각 중..."):
                    answer = ask_question(store.name, temp_history, model_name)

                if answer:
                    st.markdown(answer)
                else:
                    answer = "⚠️ 응답을 가져오지 못했습니다. (서버 과부하 또는 API 에러)"
                    st.markdown(answer)

        # history 업데이트
        st.session_state.history.append(user_msg)
        model_msg = types.Content(
            role="model",
            parts=[types.Part(text=answer)]
        )
        st.session_state.history.append(model_msg)

        # 히스토리가 너무 길어지면 뒤에서 N턴만 남기기
        if len(st.session_state.history) > MAX_TURNS * 2:
            st.session_state.history = st.session_state.history[-MAX_TURNS * 2:]
else:
    # 새 질문이 없으면 compact 렌더링만
    render_history_compact()
