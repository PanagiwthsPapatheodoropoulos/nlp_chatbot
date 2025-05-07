import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
import os
import tkinter as tk
from tkinter import scrolledtext
import time
import threading
from tkinter import scrolledtext, font
from tkinter import ttk

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

class AdvancedChatbot:
    def __init__(self, knowledge_base_file):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.knowledge_base_file = knowledge_base_file
        self.knowledge_base = self.load_knowledge_base()
        self.vectorizer = TfidfVectorizer()
        self.response_vectors = None
        self.fit()

    def load_knowledge_base(self):
        if os.path.exists(self.knowledge_base_file):
            with open(self.knowledge_base_file, 'r') as file:
                return json.load(file)
        return []

    def save_knowledge_base(self):
        with open(self.knowledge_base_file, 'w') as file:
            json.dump(self.knowledge_base, file, indent=2)

    def preprocess(self, text):
        tokens = nltk.word_tokenize(text.lower())
        return ' '.join([self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words])

    def fit(self):
        if not self.knowledge_base:
            return
        preprocessed_responses = [self.preprocess(item['input']) for item in self.knowledge_base]
        self.response_vectors = self.vectorizer.fit_transform(preprocessed_responses)

    def get_response(self, user_input):
        if not self.knowledge_base:
            return "I don't have any knowledge yet. Please teach me!"
        
        preprocessed_input = self.preprocess(user_input)
        input_vector = self.vectorizer.transform([preprocessed_input])
        
        similarities = cosine_similarity(input_vector, self.response_vectors)
        most_similar_idx = np.argmax(similarities)
        
        if similarities[0][most_similar_idx] > 0.5:
            return self.knowledge_base[most_similar_idx]['response']
        else:
            return "I'm not sure about that. Can you tell me more?"

    def learn(self, user_input, correct_response):
        self.knowledge_base.append({"input": user_input, "response": correct_response})
        self.save_knowledge_base()
        self.fit()

class ChatbotGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Advanced NLP Chatbot")
        self.master.configure(bg='#000000')
        
        window_width = 800
        window_height = 600
        screen_width = master.winfo_screenwidth()
        screen_height = master.winfo_screenheight()
        center_x = int(screen_width/2 - window_width/2)
        center_y = int(screen_height/2 - window_height/2)
        self.master.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
        
        self.chatbot = AdvancedChatbot('knowledge_base.json')
        self.feedback_mode = False
        self.last_question = None
        self.waiting_for_better_response = False
        self.asking_for_more_questions = False  # New flag to track state
        self.typing_after_id = None
        self.message_queue = []
        
        self.create_widgets()
        self.configure_grid()
        self.queue_message("Chatbot: Hello! How can I assist you today?")

    def create_widgets(self):
        self.main_frame = tk.Frame(self.master, bg='#000000')
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        self.chat_frame = tk.Frame(self.main_frame, bg='#000000')
        self.chat_frame.pack(fill=tk.BOTH, expand=True)

        # Create a custom font
        custom_font = font.Font(family="Consolas", size=12, weight="bold")

        self.chat_display = scrolledtext.ScrolledText(
            self.chat_frame,
            wrap=tk.WORD,
            bg='#1a1a1a',
            fg='#ffffff',
            font=custom_font,
            padx=10,
            pady=10
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True)
        self.chat_display.config(state=tk.DISABLED)

        self.input_frame = tk.Frame(self.main_frame, bg='#000000')
        self.input_frame.pack(fill=tk.X, pady=(20, 0))

        
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Custom.TEntry', 
                        fieldbackground='#1a1a1a', 
                        foreground='#ffffff', 
                        insertcolor='#ffffff',
                        font=custom_font,
                        padding=10,
                        borderwidth=0,
                        relief='flat',
                        bordercolor='#1a1a1a',
                        focuscolor='#1a1a1a',
                        lightcolor='#1a1a1a',
                        darkcolor='#1a1a1a')

        self.user_input = ttk.Entry(
            self.input_frame,
            style='Custom.TEntry'
        )
        
        self.user_input.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.user_input.bind("<Return>", self.send_message)
        self.user_input.insert(0, "Ask me a question")
        self.user_input.bind("<FocusIn>", self.clear_placeholder)
        self.user_input.bind("<FocusOut>", self.restore_placeholder)

       
        style.configure('Custom.TButton',
                        background='#2D2D2D',
                        foreground='#ffffff',
                        font=custom_font,
                        borderwidth=0,
                        relief='flat',
                        padding=10,
                        focuscolor='#404040',
                        focusthickness=0)

        style.map('Custom.TButton',
                  background=[('active', '#404040')],
                  foreground=[('active', '#ffffff')])


        self.send_button = tk.Button(
            self.input_frame,
            text="Send",
            bg='#2D2D2D',
            fg='#ffffff',
            activebackground='#404040',
            activeforeground='#ffffff',
            relief=tk.FLAT,
            padx=20,
            font=custom_font,
            command=self.send_message
        )
        self.send_button.pack(side=tk.LEFT, padx=(10, 0))

    def configure_grid(self):
        self.master.grid_rowconfigure(0, weight=1)
        self.master.grid_columnconfigure(0, weight=1)

    def clear_placeholder(self, event):
        if self.user_input.get() == "Ask me a question":
            self.user_input.delete(0, tk.END)
            self.user_input.config(foreground='#ffffff')

    def restore_placeholder(self, event):
        if not self.user_input.get():
            self.user_input.insert(0, "Ask me a question")
            self.user_input.config(foreground='#808080')

    def send_message(self, event=None):
        message = self.user_input.get().strip()
        if not message or message == "Ask me a question":
            return

        if self.feedback_mode:
            self.handle_feedback(message)
        elif self.asking_for_more_questions:  # Check if we're asking for more questions
            self.handle_more_questions_response(message)
        else:
            self.process_user_message(message)

    def type_next_message(self):
        if self.message_queue:
            message = self.message_queue.pop(0)
            self.type_message(message, 0)
        else:
            self.typing_after_id = None

    def queue_message(self, message):
        self.message_queue.append(message)
        if not self.typing_after_id:
            self.type_next_message()

    def process_user_message(self, message):
        self.display_message(f"You: {message}")
        self.user_input.delete(0, tk.END)

        chatbot_response = self.chatbot.get_response(message)
        self.queue_message(f"Chatbot: {chatbot_response}")

        if "I'm not sure about that" in chatbot_response:
            self.last_question = message
            self.feedback_mode = True
            self.asking_for_more_questions = False
            self.queue_message("Chatbot: Was this response helpful? (yes/no)")
        else:
            self.asking_for_more_questions = True  # Set flag to indicate we're asking for more questions
            self.queue_message("Chatbot: Do you have any more questions?")

    def handle_feedback(self, feedback):
        if feedback.lower() == 'no':
            if not self.waiting_for_better_response:
                self.queue_message("Chatbot: What would be a better response?")
                self.waiting_for_better_response = True
        elif feedback.lower() == 'yes':
            self.feedback_mode = False
            self.waiting_for_better_response = False
            self.asking_for_more_questions = True  # Now we're asking for more questions
            self.queue_message("Chatbot: Great! Do you have any more questions?")
        elif self.waiting_for_better_response:
            self.chatbot.learn(self.last_question, feedback)
            self.feedback_mode = False
            self.waiting_for_better_response = False
            self.asking_for_more_questions = True  # Now we're asking for more questions
            self.queue_message("Chatbot: Thank you for teaching me! Do you have any more questions?")
        
        self.user_input.delete(0, tk.END)

    def handle_more_questions_response(self, response):
        self.display_message(f"You: {response}")
        self.user_input.delete(0, tk.END)
        
        # Check for negative responses
        if response.lower() in ['no', 'nope', 'not now', 'no more questions']:
            # User doesn't have more questions
            self.asking_for_more_questions = False
            self.queue_message("Chatbot: I'm here whenever you need help. Have a great day!")
        # Check for positive responses
        elif response.lower() in ['yes', 'yeah', 'yep', 'sure', 'ok', 'okay']:
            # User wants to ask more questions but hasn't asked yet
            self.asking_for_more_questions = False
            self.queue_message("Chatbot: Great! What would you like to know?")
        else:
            # Treat as a new question
            self.asking_for_more_questions = False
            chatbot_response = self.chatbot.get_response(response)
            self.queue_message(f"Chatbot: {chatbot_response}")
            
            if "I'm not sure about that" in chatbot_response:
                self.last_question = response
                self.feedback_mode = True
                self.queue_message("Chatbot: Was this response helpful? (yes/no)")
            else:
                self.asking_for_more_questions = True
                self.queue_message("Chatbot: Do you have any more questions?")

    def display_message(self, message):
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, message + "\n")
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)

    def display_message_with_typing(self, message):
        self.chat_display.config(state=tk.NORMAL)
        self.type_message(message, 0)

    def type_message(self, message, index):
        if index == 0:
            self.chat_display.config(state=tk.NORMAL)
        
        if index < len(message):
            self.chat_display.insert(tk.END, message[index])
            self.chat_display.see(tk.END)
            self.typing_after_id = self.master.after(20, self.type_message, message, index + 1)
        else:
            self.chat_display.insert(tk.END, "\n")
            self.chat_display.config(state=tk.DISABLED)
            self.typing_after_id = self.master.after(500, self.type_next_message)

if __name__ == "__main__":
    root = tk.Tk()
    app = ChatbotGUI(root)
    root.mainloop()