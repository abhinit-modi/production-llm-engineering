from sqladmin import ModelView
from models import User, Message

# SQLAdmin views for database management UI
class UserAdmin(ModelView, model=User):
    column_list = ['id', 'username']

class MessageAdmin(ModelView, model=Message):
    column_list = ['id', 'text', 'type', 'timestamp', 'user_id']