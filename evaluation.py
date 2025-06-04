import random
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize
import numpy as np

# ========================
# 1️⃣ GPU Kullanım Kontrolü
# ========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Using device: {device}")

# ========================
# 2️⃣ Eğitilmiş Modeli ve Tokenizer'ı Yükleme
# ========================
model_path = "D:/YAPAY ZEKA MODÜLLER/trained_model"  # Modelin kaydedildiği dizin
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path).to(device).eval()

# Etiket eşleşmeleri (0: Corporate, 1: Human Resources, 2: Other, 3: Support)
id2label = {0: "Corporate", 1: "Human Resources", 2: "Other", 3: "Support"}

# ========================
# 3️⃣ Test Verisi Üretimi: Her kategoriden 100 farklı test metni
# ========================

def generate_test_sentences(templates, count):
    """
    Verilen şablon listesinden rastgele seçip, her biri sonuna örnek numarası ekleyerek 'count' adet farklı cümle üretir.
    """
    sentences = []
    for i in range(count):
        template = random.choice(templates)
        # Ek varyasyon için örnek numarasını ekliyoruz
        sentence = f"{template} (example {i+1})"
        sentences.append(sentence)
    return sentences

# Farklı şablonlar tanımlıyoruz:
corporate_templates = [
    "Could you please send me the invoice for my last purchase?",
    "I need a copy of my invoice for my recent corporate expense.",
    "Please forward the invoice related to my business order."
]

hr_templates = [
    "I would like to inquire about job opportunities in your HR department.",
    "Could you provide more details on human resources openings?",
    "I am interested in pursuing a career in HR at your company."
]

other_templates = [
    "I have a query that does not seem to fit into any of the usual categories.",
    "My question is miscellaneous and cannot be classified under standard labels.",
    "I need information on a matter that is not related to corporate, HR, or support."
]

support_templates = [
    "I need support regarding my recent order.",
    "I require assistance with an issue in my order delivery.",
    "My order has encountered a problem; please help as soon as possible."
]

# Her kategori için 100 farklı örnek oluşturuyoruz.
corporate_texts = generate_test_sentences(corporate_templates, 100)
hr_texts = generate_test_sentences(hr_templates, 100)
other_texts = generate_test_sentences(other_templates, 100)
support_texts = generate_test_sentences(support_templates, 100)

# Tüm metinleri birleştiriyoruz: toplam 400 örnek
test_texts = corporate_texts + hr_texts + other_texts + support_texts
# Gerçek etiketler: önce 100 adet 0, ardından 100 adet 1, 100 adet 2, son olarak 100 adet 3
test_labels = [0]*100 + [1]*100 + [2]*100 + [3]*100

# ========================
# 4️⃣ Model Tahminlerinin Hesaplanması
# ========================
# Test metinlerini tokenleştir
inputs = tokenizer(test_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
inputs = {key: value.to(device) for key, value in inputs.items()}  # GPU kullanımı için taşıma

with torch.no_grad():
    outputs = model(**inputs)

# Tahmin edilen sınıf id'leri
predictions = torch.argmax(outputs.logits, dim=1).cpu().numpy()
# ROC-AUC için, her sınıfa ait olasılık skorlarını elde ediyoruz (softmax uygulanmış çıktı)
y_score = torch.nn.functional.softmax(outputs.logits, dim=1).cpu().numpy()

print(f"📌 Gerçek etiketler (ilk 20 örnek): {test_labels[:20]} ... (toplam {len(test_labels)} örnek)")
print(f"📌 Model tahminleri (ilk 20 örnek): {predictions[:20]} ... (toplam {len(predictions)} örnek)")

# ========================
# 5️⃣ Performans Metriklerinin Hesaplanması
# ========================
# Accuracy hesaplanması
accuracy = accuracy_score(test_labels, predictions)
print(f"🔹 Accuracy: {accuracy:.4f}")

# F1-Score hesaplanması (weighted)
f1 = f1_score(test_labels, predictions, average="weighted")
print(f"🔹 F1-Score: {f1:.4f}")

# Confusion Matrix oluşturulması
conf_matrix = confusion_matrix(test_labels, predictions)
print("🔹 Confusion Matrix:")
print(conf_matrix)

# ROC-AUC Hesaplaması
unique_classes = np.unique(test_labels)
if len(unique_classes) < 2:
    print("🔹 ROC-AUC Score: Hesaplanamıyor. Test etiketlerinde 2'den az sınıf mevcut.")
else:
    # Gerçek etiketleri, test setinde bulunan sınıflara göre binarize edelim
    y_true_bin = label_binarize(test_labels, classes=unique_classes)
    # Modelin çıkışındaki olasılıkların da test setinde bulunan sınıflara göre sütunlarını seçelim
    y_score_reduced = y_score[:, unique_classes]
    try:
        roc_auc = roc_auc_score(y_true_bin, y_score_reduced, multi_class="ovo")
        print(f"🔹 ROC-AUC Score: {roc_auc:.4f}")
    except Exception as e:
        print("🔹 ROC-AUC hesaplanırken hata oluştu:", e)
