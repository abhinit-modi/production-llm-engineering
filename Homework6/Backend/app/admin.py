from sqladmin import ModelView
from models import User, Message

# TODO: add the column you want to see in the admin UI
class UserAdmin(ModelView, model=User):
    column_list = ['id', 'username']

class MessageAdmin(ModelView, model=Message):
    column_list = ['id', 'text', 'type', 'timestamp', 'user_id']