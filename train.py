# train.py

# ========================
# ADIM 0: Gerekli Kütüphanelerin İçe Aktarılması
# ========================
import torch
import re
from collections import Counter
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)
from datasets import load_dataset
from deep_translator import GoogleTranslator
from langdetect import detect

# GPU kontrolü
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Using device: {device}")

# ========================
# ADIM 1: CSV Dosyasını Yükleme ve Ön İşleme
# ========================
csv_file = "D:/enron_email_dataset.csv"
raw_dataset = load_dataset("csv", data_files=csv_file, split="train")
print("CSV dosyasındaki sütunlar:", raw_dataset.column_names)

# ========================
# ADIM 2: Sütunları Düzenleme ve Etiketleme
# ========================
raw_dataset = raw_dataset.rename_column("message", "text")

def extract_label(example):
    parts = example["file"].split('/')
    if len(parts) >= 2:
        return {"raw_label": parts[-2]}
    return {"raw_label": "Other"}

dataset = raw_dataset.map(extract_label)

def map_to_high_level(example):
    raw_label = example["raw_label"].lower()
    text = example["text"].lower()
    
    if "sent" in raw_label:
        candidate = "Corporate"
    elif "inbox" in raw_label:
        candidate = "Support"
    elif "draft" in raw_label or "spam" in raw_label:
        candidate = "Other"
    elif "hr" in raw_label or "staff" in raw_label or "people" in raw_label:
        candidate = "Human Resources"
    else:
        candidate = "Other"
    
    if "invoice" in text or "purchase" in text:
        candidate = "Corporate"
    elif "support" in text or "help" in text or "order" in text:
        candidate = "Support"
    elif "hr" in text or "staff" in text:
        candidate = "Human Resources"
    
    return {"label": candidate}

dataset = dataset.map(map_to_high_level)
print("Oluşturulan yüksek seviyeli etiketler:", set(dataset["label"]))

# ========================
# ADIM 3: Metin Temizliği
# ========================
def clean_text(example):
    t = example["text"]
    t = re.sub(r'<.*?>', ' ', t)
    t = re.sub(r'http\S+|www\S+|https\S+', ' ', t, flags=re.MULTILINE)
    t = re.sub(r'\S+@\S+', ' ', t)
    t = re.sub(r'^(from|sent|to|subject):.*$', ' ', t, flags=re.IGNORECASE | re.MULTILINE)
    t = re.sub(r'\s+', ' ', t).strip()
    return {"text": t}

dataset = dataset.map(clean_text)
dataset = dataset.remove_columns(["file", "raw_label"])

# ========================
# ADIM 4: Etiket Sayısallaştırma
# ========================
unified_labels = ["Corporate", "Human Resources", "Other", "Support"]
label2id = {label: idx for idx, label in enumerate(unified_labels)}
id2label = {idx: label for label, idx in label2id.items()}

def convert_label(example):
    example["label"] = label2id.get(example["label"], label2id["Other"])
    return example

dataset = dataset.map(convert_label)
print("Sayısallaştırılmış etiket dağılımı:", Counter(dataset["label"]))

# ========================
# ADIM 5: Train-Test Split (Demosal eğitim için %50 test)
# ========================
split_dataset = dataset.train_test_split(test_size=0.5, seed=42)
train_dataset = split_dataset["train"]
test_dataset = split_dataset["test"]

# ========================
# ADIM 6: Tokenizasyon
# ========================
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
tokenizer.model_max_length = 512

def tokenize_fn(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

tokenized_train_dataset = train_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
tokenized_test_dataset = test_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

# ========================
# ADIM 7: Model Eğitimi
# ========================
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    fp16=True,
    weight_decay=0.08,
    logging_dir="./logs",
    logging_steps=10,
    report_to=[],
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss"
)

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(unified_labels),
    id2label=id2label,
    label2id=label2id
).to(device)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_test_dataset,
    tokenizer=tokenizer
)

trainer.train()

# ========================
# ADIM 8: Şablon Tabanlı Yanıt Sözlüğü
# ========================
TEMPLATE_RESPONSES = {
    "Corporate": "Your request has been received. Our relevant department will get back to you as soon as possible.",
    "Human Resources": "Our Human Resources team has reviewed your inquiry and will contact you shortly. Thank you for your interest.",
    "Other": "We have received your message. It will be reviewed and processed accordingly.",
    "Support": "Our support team is evaluating your request and will reach out to you as soon as possible."
}

# ========================
# ADIM 9: Eğitilmiş Modeli ve Tokenizer'ı Kaydetme
# ========================
save_path = "D:/YAPAY ZEKA MODÜLLER/trained_model"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print("🚀 Model eğitimi başarıyla tamamlandı ve model kaydedildi!")
