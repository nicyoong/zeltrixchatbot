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