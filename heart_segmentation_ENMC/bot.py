
import requests
import os
from dotenv import load_dotenv

load_dotenv()

class Bot():
    def __init__(self):
        self.bot_token = os.environ['botAPI']
        self.url = "https://api.telegram.org/bot%s"%(self.bot_token) + "/"
        self.ID = [37216747]

    def send_message(self, message_text):
        chat_id = self.ID[0]
        params = {"chat_id": chat_id,
                    "text": message_text}
        response = requests.post(self.url + "sendMessage", data=params)
        return response

