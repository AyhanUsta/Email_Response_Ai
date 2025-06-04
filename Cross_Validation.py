import re
import torch
import numpy as np
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
from datasets import load_dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize

# ========================
# 1️⃣ Veri Setinin Yüklenmesi ve Ön İşleme
# ========================

# Enron veri setini CSV dosyasından yüklüyoruz (dosya yolunu güncelleyin)
csv_file = "D:/enron_email_dataset.csv"
dataset = load_dataset("csv", data_files=csv_file, split="train")

# "message" sütununu "text" olarak yeniden adlandırıyoruz
dataset = dataset.rename_column("message", "text")

# Dosya yolundaki bilgiye göre etiketi çıkaran fonksiyon
def extract_label(example):
    parts = example["file"].split('/')
    if len(parts) >= 2:
        return {"raw_label": parts[-2]}
    return {"raw_label": "Other"}

dataset = dataset.map(extract_label)

# E-posta metnine ve dosya yolundaki bilgilere göre yüksek seviyeli etiket ataması
def map_to_high_level(example):
    raw_label = example["raw_label"].lower()
    text = example["text"].lower()
    
    # Dosya yolundan gelen ipuçları
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
    
    # Metin içeriğinden gelen ipuçlarıyla ek düzenleme:
    if "invoice" in text or "purchase" in text:
        candidate = "Corporate"
    elif "support" in text or "help" in text or "order" in text:
        candidate = "Support"
    elif "hr" in text or "staff" in text:
        candidate = "Human Resources"
    
    return {"label": candidate}

dataset = dataset.map(map_to_high_level)

# Metin temizliği: HTML tag, URL, e-mail adres, gereksiz boşluklar temizleniyor
def clean_text(example):
    t = example["text"]
    t = re.sub(r'<.*?>', ' ', t)
    t = re.sub(r'http\S+|www\S+|https\S+', ' ', t, flags=re.MULTILINE)
    t = re.sub(r'\S+@\S+', ' ', t)
    t = re.sub(r'^(from|sent|to|subject):.*$', ' ', t, flags=re.IGNORECASE | re.MULTILINE)
    t = re.sub(r'\s+', ' ', t).strip()
    return {"text": t}

dataset = dataset.map(clean_text)

# Dosya yoluna ait bilgiler ("file", "raw_label") artık gerekmiyor
dataset = dataset.remove_columns(["file", "raw_label"])

# Etiketleri sayısal forma dönüştürüyoruz:
unified_labels = ["Corporate", "Human Resources", "Other", "Support"]
label2id = {label: idx for idx, label in enumerate(unified_labels)}
id2label = {idx: label for label, idx in label2id.items()}

def convert_label(example):
    example["label"] = label2id.get(example["label"], label2id["Other"])
    return example

dataset = dataset.map(convert_label)

# Tüm örneklerin metinlerini ve etiketlerini alıyoruz:
texts = dataset["text"]
labels = dataset["label"]

print(f"Dataset içerisindeki örnek sayısı: {len(texts)}")

# ========================
# 2️⃣ Model ve Tokenizer Yüklemesi
# ========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model_path = "D:/YAPAY ZEKA MODÜLLER/trained_model"  # Eğitilmiş modelin bulunduğu yol
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path).to(device).eval()

# ========================
# 3️⃣ Cross Validation Uygulaması (StratifiedKFold + Batch İşleme)
# ========================
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

accuracy_list = []
f1_list = []
roc_auc_list = []
confusion_total = np.zeros((len(unified_labels), len(unified_labels)), dtype=int)

labels_arr = np.array(labels)
batch_size = 16  # İhtiyacınıza göre batch boyutunu ayarlayın

fold_num = 1
for _, test_idx in skf.split(texts, labels_arr):
    # Her fold için test verisini alıyoruz
    fold_texts = [texts[i] for i in test_idx]
    fold_labels = [labels[i] for i in test_idx]

    all_predictions = []
    all_y_scores = []

    # Batch işlemi ile GPU belleğini verimli kullanıyoruz
    for i in range(0, len(fold_texts), batch_size):
        batch_texts = fold_texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        batch_preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        batch_y_score = torch.nn.functional.softmax(outputs.logits, dim=1).cpu().numpy()
        all_predictions.append(batch_preds)
        all_y_scores.append(batch_y_score)

    # Tüm batch'lerden elde edilen sonuçları birleştiriyoruz
    all_predictions = np.concatenate(all_predictions)
    all_y_scores = np.concatenate(all_y_scores)

    acc = accuracy_score(fold_labels, all_predictions)
    f1_val = f1_score(fold_labels, all_predictions, average="weighted")
    conf_matrix = confusion_matrix(fold_labels, all_predictions, labels=range(len(unified_labels)))
    confusion_total += conf_matrix

    # ROC-AUC hesaplaması için, gerçek etiketleri binarize ediyoruz
    y_true_bin = label_binarize(fold_labels, classes=range(len(unified_labels)))
    try:
        roc_auc = roc_auc_score(y_true_bin, all_y_scores, multi_class="ovo")
    except Exception as e:
        roc_auc = float('nan')
    
    print(f"--- Fold {fold_num} ---")
    print(f"Accuracy: {acc:.4f}, F1-Score: {f1_val:.4f}, ROC-AUC: {roc_auc:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("")
    
    accuracy_list.append(acc)
    f1_list.append(f1_val)
    roc_auc_list.append(roc_auc)
    fold_num += 1

avg_accuracy = np.mean(accuracy_list)
avg_f1 = np.mean(f1_list)
avg_roc_auc = np.mean([x for x in roc_auc_list if not np.isnan(x)])

print("=== Genel Cross Validation Sonuçları ===")
print(f"Average Accuracy: {avg_accuracy:.4f}")
print(f"Average F1-Score: {avg_f1:.4f}")
print(f"Average ROC-AUC: {avg_roc_auc:.4f}")
print("Toplam Confusion Matrix (Toplam fold bazında):")
print(confusion_total)
