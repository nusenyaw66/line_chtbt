from flask import Flask, request, abort
from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import Configuration, ApiClient, MessagingApi, ReplyMessageRequest, TextMessage
from linebot.v3.webhooks import MessageEvent, TextMessageContent
from qa_chatbot import qa_line_chatbot
# from qa_chatbot_local_llm import qa_line_chatbot

from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")

# Initialize Flask app
app = Flask(__name__)

# Initialize LINE Configuration and Messaging API
configuration = Configuration(access_token=CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(channel_secret=CHANNEL_SECRET)

# Webhook endpoint
@app.route("/callback", methods=['POST'])
def callback():
    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']

    # get request body as text
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    # handle webhook body
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        app.logger.info("Invalid signature. Please check your channel access token/channel secret.")
        abort(400)

    return 'OK'

# Handle text messages
@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event):
    user_message = event.message.text
    
    # Call your RAG chatbot to process the message
    response = qa_line_chatbot(user_message)

    # Send response back to LINE using ApiClient
    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        line_bot_api.reply_message_with_http_info(
            ReplyMessageRequest(
                replyToken=event.reply_token,
                messages=[TextMessage(text=response)]
            )
        )

if __name__ == "__main__":
    app.run(debug=True, port=4500)