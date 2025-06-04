# predict.py

import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
from deep_translator import GoogleTranslator
from langdetect import detect

# ========================
# ADIM 0: Gerekli Kütüphanelerin İçe Aktarılması ve GPU Kontrolü
# ========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Using device: {device}")

# ========================
# ADIM 1: Kaydedilmiş Modeli ve Tokenizer'ı Yükleme
# ========================
# Modelin kaydedildiği dizini belirtin (tam yol kullanmak daha sağlıklı olabilir)
model_path = "D:/YAPAY ZEKA MODÜLLER/trained_model"
model = DistilBertForSequenceClassification.from_pretrained(model_path).to(device)
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)

# Etiket sözlüğü (model eğitimi sırasında kullanılan etiketlerin sırası)
id2label = {0: "Corporate", 1: "Human Resources", 2: "Other", 3: "Support"}

# Şablon yanıtlar
TEMPLATE_RESPONSES = {
    "Corporate": "Your request has been received. Our relevant department will get back to you as soon as possible.",
    "Human Resources": "Our Human Resources team has reviewed your inquiry and will contact you shortly. Thank you for your interest.",
    "Other": "We have received your message. It will be reviewed and processed accordingly.",
    "Support": "Our support team is evaluating your request and will reach out to you as soon as possible."
}

# ========================
# ADIM 2: process_email() Fonksiyonu (Inference)
# ========================
def process_email(email_text):
    # Gelen e-posta metninin dilini algıla
    src_lang = detect(email_text)
    
    # E-posta metnini İngilizceye çevir (gerekirse)
    translated_text = email_text if src_lang == "en" else GoogleTranslator(source=src_lang, target="en").translate(email_text)
    
    # Model girdi hazırlığı: Tokenizasyon, truncation, padding
    inputs = tokenizer(translated_text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    predicted_idx = outputs.logits.argmax(dim=-1).item()
    predicted_label = id2label[predicted_idx]
    response_en = TEMPLATE_RESPONSES[predicted_label]
    
    # Yanıtı, eğer gerekli ise, orijinal dile çevir
    final_response = response_en if src_lang == "en" else GoogleTranslator(source="en", target=src_lang).translate(response_en)
    return predicted_label, response_en, final_response

# ========================
# ADIM 3: Test ve Kullanıcı Girdisi
# ========================
if __name__ == "__main__":
    # Test e-postaları (farklı dillerde)
    test_emails = [
        "I need support regarding my latest order.",
        "Can you send me the invoice for my last purchase?",
        "Siparişimle ilgili destek istiyorum.",
        "Son satın alma işlemimle ilgili faturayı gönderebilir misiniz?",
        "Quiero cancelar mi pedido.",
        "Comment puis-je obtenir de l'aide avec mon achat récent ?",
    ]
    
    print("\n--- Model Çıktılarını Test Ediyoruz ---\n")
    for email in test_emails:
        label, response_en, response_final = process_email(email)
        print(f"📩 E-posta: {email}")
        print(f"🏷 Tahmin Edilen Kategori: {label}")
        print(f"💬 İngilizce Yanıt: {response_en}")
        print(f"🌍 Orijinal Dilinde Yanıt: {response_final}\n")
    
    # Alternatif: Kullanıcıdan e-posta metni alarak tahmin yap
    user_input = input("Lütfen e-posta metnini giriniz: ")
    label, response_en, response_final = process_email(user_input)
    print(f"\nTahmin Edilen Kategori: {label}")
    print(f"Yanıt: {response_final}")
