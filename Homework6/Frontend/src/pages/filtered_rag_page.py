from streamlit.runtime.scriptrunner import get_script_run_ctx
from chat import chat_interface

chat_title = "Filtered Rag Chat App"
url = "https://damienbenveniste-backend-hf.hf.space/filtered_rag/stream"
page_hash = get_script_run_ctx().page_script_hash

chat_interface(chat_title, page_hash, url)