from streamlit.runtime.scriptrunner import get_script_run_ctx
from chat import chat_interface

chat_title = "Rag Chat App"
url = "https://damienbenveniste-backend-hf.hf.space/rag/stream"
page_hash = get_script_run_ctx().page_script_hash

chat_interface(chat_title, page_hash, url)