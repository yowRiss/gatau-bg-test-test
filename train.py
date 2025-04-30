import os
import json
from dotenv import load_dotenv
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from sentence_transformers import SentenceTransformer, util

# Load .env
load_dotenv()

# Google Sheets authentication
google_creds_path = os.getenv("GOOGLE_CREDENTIALS_PATH")
with open(google_creds_path) as f:
    json_creds = json.load(f)

scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_dict(json_creds, scope)
client = gspread.authorize(creds)

# Buka Google Sheet
spreadsheet = client.open_by_key(os.getenv("GOOGLE_SHEET_ID"))
sheet = spreadsheet.worksheet(os.getenv("GOOGLE_SHEET_TAB"))

# Ambil data awal untuk mode jawaban
data = sheet.get_all_records(expected_headers=["Question", "Answer"])
questions = [row['Question'] for row in data]
answers = [row['Answer'] for row in data]

# Load model
model = SentenceTransformer('distilbert-base-nli-mean-tokens')
embeddings = model.encode(questions)

def jawab_ai(user_input):
    user_embedding = model.encode([user_input])
    scores = util.cos_sim(user_embedding, embeddings)[0]
    best_match = scores.argmax()
    confidence = scores[best_match]

    if confidence > 0.6:
        return answers[best_match]
    else:
        return None

def train_manual():
    question = input("Masukkan pertanyaan baru: ").strip()
    answer = input("Masukkan jawabannya: ").strip()
    if question and answer:
        sheet.append_row([question, answer])
        print("✅ Data baru berhasil ditambahkan ke Google Sheet.")
    else:
        print("❌ Pertanyaan dan jawaban tidak boleh kosong.")


if __name__ == "__main__":
    print("Ketik `train` untuk menambahkan data manual.")
    print("Ketik `exit` untuk keluar.\n")

    while True:
        q = input("Tanya > ").strip()
        if q.lower() in ["exit", "quit"]:
            break
        elif q.lower() == "train":
            train_manual()
        else:
            jawaban = jawab_ai(q)
            if jawaban:
                print("Jawab >", jawaban)
            else:
                print("Saya belum tahu jawabannya.")
