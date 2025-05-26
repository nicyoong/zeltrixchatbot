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

    def get_response(self, user_id, user_input, is_reminder=False):
        try:
            # Enforce global rate limit first
            self._enforce_rate_limit()
            self.request_timestamps.append(time.time())

            # Handle user context
            if user_id not in self.user_contexts:
                self.user_contexts[user_id] = {
                    'conversation_history': [],
                    'current_tokens': 0,
                    'last_activity': time.time(),
                    'reminder_sent': False
                }
            uc = self.user_contexts[user_id]

            # Update activity and reset reminder flag for real user messages
            if not is_reminder:
                uc['last_activity'] = time.time()
                uc['reminder_sent'] = False

            # Add user message
            uc['conversation_history'].append({"role": "user", "content": user_input})
            uc['current_tokens'] += self._calculate_tokens(user_input)

            print(f"User ID: {user_id}")
            print(f"Token Count for current User ID: {uc['current_tokens']}")

            # Get API response
            response = self.client.chat.completions.create(
                model=self.model,
                messages=uc['conversation_history']
            )
            ai_response = response.choices[0].message.content
            
            # Add AI response
            uc['conversation_history'].append({"role": "assistant", "content": ai_response})
            uc['current_tokens'] += self._calculate_tokens(ai_response)

            # Truncate history
            self._truncate_history(uc)

            return ai_response

        except Exception as e:
            return f"⚠️ Error: {str(e)}"
