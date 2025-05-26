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

    # def _is_ascii_word(self, word):
    #     return all(ord(c) <= 127 for c in word)

    # def _calculate_tokens(self, text):
    #     total = 0.0
    #     for word in text.split():
    #         if self._is_ascii_word(word):
    #             total += 1.3
    #         else:
    #             total += len(word) * 1.5
    #     return math.ceil(total)
    
    def _calculate_tokens(self, text):
        """Count tokens using GPT-4's actual tokenization"""
        return len(self.encoder.encode(text))
    
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
        
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    chatbot = context.bot_data['chatbot']

    async def keep_typing():
        """Show typing action with random intervals"""
        try:
            while True:
                await context.bot.send_chat_action(
                    chat_id=user.id,
                    action=constants.ChatAction.TYPING
                )
                wait_time = random.uniform(1.5, 4.5)
                await asyncio.sleep(wait_time)
        except asyncio.CancelledError:
            print(f"    ↳ Typing stopped for {user.id}")
    
    # Start typing task and processing task concurrently
    typing_task = asyncio.create_task(keep_typing())

    try:
        # Get response from API
        loop = asyncio.get_event_loop()
        response_text = await loop.run_in_executor(
            None, 
            chatbot.get_response, 
            user.id, 
            update.message.text
        )
        
        tokens = chatbot._calculate_tokens(response_text)
        typing_delay = chatbot._calculate_typing_delay(response_text)
        
        print(f"\n⌨️ Response ready in {typing_delay:.1f}s | {tokens} tokens")
        
        await asyncio.sleep(typing_delay)
        
        # Send message and IMMEDIATELY cancel typing
        await update.message.reply_text(response_text)
        typing_task.cancel()
        
    except Exception as e:
        typing_task.cancel()
        await update.message.reply_text(f"⚠️ Error: {str(e)}")

    # Ensure clean task cancellation
    try:
        await typing_task
    except asyncio.CancelledError:
        pass

async def check_inactive_users(context: ContextTypes.DEFAULT_TYPE):
    """Periodic job to check for inactive users"""
    chatbot = context.bot_data['chatbot']
    current_time = time.time()
    
    # Iterate over a copy to avoid dictionary changed during iteration issues
    for user_id, user_context in chatbot.user_contexts.copy().items():
        try:
            # Skip if context structure is incomplete
            if 'last_activity' not in user_context or 'reminder_sent' not in user_context:
                continue
                
            last_active = user_context['last_activity']
            needs_reminder = (
                (current_time - last_active) >= 3600 and  # 1 hour
                not user_context['reminder_sent']
            )
            
            if needs_reminder:
                # Generate reminder through the API
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    chatbot.get_response,
                    user_id,
                    "Continue the last text",
                    True  # Mark as reminder to prevent activity update
                )
                
                # Send reminder and update flag
                await context.bot.send_message(chat_id=user_id, text=response)
                user_context['reminder_sent'] = True
                print(f"Sent reminder to {user_id}")
                
        except Exception as e:
            print(f"Reminder error for {user_id}: {str(e)}")

def main():
    chatbot = ShapeChatBot()
    app = Application.builder().token(os.getenv("ZT_TELEGRAM_BOT_TOKEN")).build()
    app.bot_data['chatbot'] = chatbot
    
    # Set up periodic check every 10 minutes
    job_queue = app.job_queue
    job_queue.run_repeating(check_inactive_users, interval=600, first=0)
    
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    print("Zeltrix Bot is running with inactivity reminders...")
    app.run_polling()

if __name__ == "__main__":
    main()
