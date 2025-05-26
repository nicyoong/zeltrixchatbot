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

    def _enforce_rate_limit(self):
        """Ensure we don't exceed 5 requests per minute (global limit)"""
        now = time.time()
        # Clean up old requests
        self.request_timestamps = [t for t in self.request_timestamps if now - t < 60]
        
        # Check if we need to wait
        while len(self.request_timestamps) >= self.rate_limit:
            oldest = self.request_timestamps[0]
            required_wait = oldest + 60 - now + 0.1  # Small buffer
            
            if required_wait > 0:
                print(f"⚠️ Rate limit exceeded. Waiting {required_wait:.1f} seconds...")
                time.sleep(required_wait)
            
            # Update tracking after waiting
            now = time.time()
            self.request_timestamps = [t for t in self.request_timestamps if now - t < 60]

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
    
    def _truncate_history(self, user_context):
        while (len(user_context['conversation_history']) > self.max_messages or 
               user_context['current_tokens'] > self.max_tokens):
            if not user_context['conversation_history']:
                break
            removed = user_context['conversation_history'].pop(0)
            user_context['current_tokens'] -= self._calculate_tokens(removed["content"])
