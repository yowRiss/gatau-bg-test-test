import os
import json
from dotenv import load_dotenv
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from sentence_transformers import SentenceTransformer, util

# Load .env
load_dotenv()


google_creds_path = os.getenv("GOOGLE_CREDENTIALS_PATH")
with open(google_creds_path) as f:
    json_creds = json.load(f)

scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_dict(json_creds, scope)
client = gspread.authorize(creds)

spreadsheet = client.open_by_key(os.getenv("GOOGLE_SHEET_ID"))
sheet = spreadsheet.worksheet(os.getenv("GOOGLE_SHEET_TAB"))


model = SentenceTransformer('distilbert-base-nli-mean-tokens')


def load_data_and_encode():
    data = sheet.get_all_records()
    questions = [row['Question'] for row in data]
    answers = [row['Answer'] for row in data]
    
    embeddings = model.encode(questions)
    return questions, answers, embeddings

questions, answers, embeddings = load_data_and_encode()


def jawab_ai(user_input, top_n=3, threshold=0.6):

    user_embedding = model.encode([user_input])
    scores = util.cos_sim(user_embedding, embeddings)[0]


    top_results = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_n]
    valid_results = [(i, s) for i, s in top_results if s > threshold]

    if valid_results:
        best_index = valid_results[0][0]
        return answers[best_index]
    else:
        print("Tidak ada jawaban yang cocok.")
        sheet.append_row([user_input, ""]) 
        global questions, answers, embeddings
        questions.append(user_input)
        answers.append("") 
        embeddings = model.encode(questions)  
        return "Saya belum tahu jawabannya, tapi sudah saya catat ya!"

# Contoh interaktif
if __name__ == "__main__":
    while True:
        q = input("Tanya > ")
        if q.lower() in ["exit", "quit"]:
            break
        print("Jawab >", jawab_ai(q))
