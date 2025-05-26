import random
from openai import OpenAI
import os
import math
import time
from dotenv import load_dotenv
from telegram import Update, constants
from telegram.ext import Application, MessageHandler, filters, ContextTypes
import asyncio
import tiktoken

load_dotenv()

class ShapeChatBot:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("SHAPES_API_KEY"),
            base_url="https://api.shapes.inc/v1/"
        )
        self.model = os.getenv("ZT_SHAPES_URL")
        self.max_messages = 25
        self.max_tokens = 65000
        self.user_contexts = {}
        
        # Rate limiting configuration
        self.rate_limit = 5  # 5 requests per minute
        self.request_timestamps = []

        self.encoder = tiktoken.get_encoding("cl100k_base")

    def _is_ascii_word(self, word):
        return all(ord(c) <= 127 for c in word)

    def _calculate_tokens(self, text):
        total = 0.0
        for word in text.split():
            if self._is_ascii_word(word):
                total += 1.3
            else:
                total += len(word) * 1.5
        return math.ceil(total)
