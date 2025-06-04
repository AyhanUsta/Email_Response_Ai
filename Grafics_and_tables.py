import random
import numpy as np
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns

# ========================
# 1️⃣ Cihaz Ayarı
# ========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Using device: {device}")

# ========================
# 2️⃣ Model ve Tokenizer Yükle
# ========================
model_path = "D:/YAPAY ZEKA MODÜLLER/trained_model"
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path).to(device).eval()

id2label = {0: "Corporate", 1: "Human Resources", 2: "Other", 3: "Support"}
class_names = [id2label[i] for i in range(4)]

# ========================
# 3️⃣ Test Cümlelerini Üret
# ========================
def generate_test_sentences(templates, count):
    return [f"{random.choice(templates)} (example {i+1})" for i in range(count)]

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

corporate_texts = generate_test_sentences(corporate_templates, 100)
hr_texts = generate_test_sentences(hr_templates, 100)
other_texts = generate_test_sentences(other_templates, 100)
support_texts = generate_test_sentences(support_templates, 100)

test_texts = corporate_texts + hr_texts + other_texts + support_texts
test_labels = [0]*100 + [1]*100 + [2]*100 + [3]*100

# ========================
# 4️⃣ Model Tahminleri
# ========================
inputs = tokenizer(test_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
inputs = {k: v.to(device) for k, v in inputs.items()}
with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits
predictions = torch.argmax(logits, dim=1).cpu().numpy()
y_score = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()

# ========================
# 5️⃣ Metrik Hesaplamaları
# ========================
accuracy = accuracy_score(test_labels, predictions)
f1 = f1_score(test_labels, predictions, average="weighted")
conf_matrix = confusion_matrix(test_labels, predictions)

y_true_bin = label_binarize(test_labels, classes=[0,1,2,3])

# ROC AUC skoru
roc_auc = roc_auc_score(y_true_bin, y_score, multi_class="ovo")

# ========================
# 6️⃣ Confusion Matrix (Isı Haritası)
# ========================
plt.figure(figsize=(6,5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# ========================
# 7️⃣ ROC Eğrileri
# ========================
plt.figure(figsize=(7,6))
for i in range(4):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
    plt.plot(fpr, tpr, label=f"{id2label[i]} (AUC = {roc_auc_score(y_true_bin[:, i], y_score[:, i]):.2f})")

plt.plot([0, 1], [0, 1], "k--")
plt.title("ROC Curves")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()
plt.show()

# ========================
# 8️⃣ Accuracy ve F1 Score Grafik
# ========================
plt.figure(figsize=(5,4))
plt.bar(["Accuracy", "F1 Score"], [accuracy, f1], color=["skyblue", "salmon"])
plt.ylim(0, 1)
plt.title("Accuracy & F1 Score")
plt.tight_layout()
plt.show()
