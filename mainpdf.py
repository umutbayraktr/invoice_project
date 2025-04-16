import json
import re
import os
from PIL import Image
import pytesseract
from pdf2image import convert_from_path

from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama

# OCR yapma fonksiyonu (resim için)
def ocr_yap(image_path, dil='tur'):
    try:
        image = Image.open(image_path)
        metin = pytesseract.image_to_string(image, lang=dil)
        return metin
    except Exception as e:
        return f"Hata oluştu: {e}"

# PDF kontrolü
def is_pdf(file_path):
    return file_path.lower().endswith(".pdf")

# PDF'den OCR metni çıkar
def extract_text_from_pdf(pdf_path, dil='tur'):
    try:
        images = convert_from_path(pdf_path, dpi=300)
        all_text = ""
        for idx, image in enumerate(images):
            metin = pytesseract.image_to_string(image, lang=dil)
            all_text += f"\n--- Sayfa {idx + 1} ---\n{metin}\n"
        return all_text
    except Exception as e:
        return f"PDF işleme hatası: {e}"

# Resimden OCR metni çıkar
def extract_text_from_image(image_path):
    return ocr_yap(image_path)

# LangChain prompt
invoice_extraction_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Sen bir fatura/fiş analiz aracı olarak çalışıyorsun. Görevin, verilen OCR çıktısından bilgileri doğru ve eksiksiz bir şekilde yapılandırılmış JSON formatında çıkartmak. Her alan zorunludur. Eğer bilgi eksikse, değer boş string (\"\"), null, 0 ya da 0.0 olmalıdır. Sadece JSON çıktısını ver, başka hiçbir açıklama ekleme."
    ),
    (
        "human",
        "OCR Metni:\n{invoice_text}\n\n"
        "Talimatlar:\n"
        "1. Aşağıdaki alanları OCR metninden çıkart:\n"
        "   - Mağaza bilgisi (unvan, adres, fiş numarası, tarih)\n"
        "   - Müşteri bilgisi (isim)\n"
        "   - Ürün kalemleri (ürün adı, kodu, miktar, birim fiyat, satır toplamı)\n"
        "   - Ödeme bilgileri (ara toplam, toplam vergi oranı, toplam vergi tutarı, yuvarlama, genel toplam, ödenen tutar, para üstü)\n\n"
        "2. JSON çıktısı şu formatta olmalı:\n"
        "{{\n"
        "  \"magazaBilgisi\": {{\n"
        "    \"unvan\": \"\",\n"
        "    \"adres\": \"\",\n"
        "    \"fisNumarasi\": \"\",\n"
        "    \"tarih\": \"\"\n"
        "  }},\n"
        "  \"musteriBilgisi\": {{\n"
        "    \"isim\": \"\"\n"
        "  }},\n"
        "  \"urunKalemleri\": [\n"
        "    {{\n"
        "      \"urunAdi\": \"\",\n"
        "      \"urunKodu\": \"\",\n"
        "      \"miktar\": 0,\n"
        "      \"birimFiyat\": 0.0,\n"
        "      \"satirToplam\": 0.0\n"
        "    }}\n"
        "  ],\n"
        "  \"odemeBilgileri\": {{\n"
        "    \"araToplam\": 0.0,\n"
        "    \"vergiOrani\": 0.0,\n"
        "    \"vergiTutari\": 0.0,\n"
        "    \"yuvarlama\": 0.0,\n"
        "    \"genelToplam\": 0.0,\n"
        "    \"odenenTutar\": 0.0,\n"
        "    \"paraUstu\": 0.0\n"
        "  }}\n"
        "}}"
    )
])

def parse_invoice_with_llm(text):
    try:
        llm = Ollama(model="llama2")
    except Exception as e:
        print(f"Hata: Ollama başlatılamadı. Detay: {e}")
        return None

    chain = invoice_extraction_prompt | llm

    try:
        result = chain.invoke({"invoice_text": text})
    except Exception as e:
        print(f"Hata: LLM zinciri çalıştırılırken sorun oluştu: {e}")
        print("--- Gönderilen Metin ---")
        print(text[:500] + "..." if len(text) > 500 else text)
        return None

    print("\nLLM'den Gelen Cevap:\n", result)

    json_match = re.search(r'\{.*\}', result, re.DOTALL)

    if json_match:
        json_str = json_match.group(0)
        try:
            parsed_result = json.loads(json_str)
            return parsed_result
        except json.JSONDecodeError as e:
            print(f"\nJSONDecodeError: {e}")
            print("Hatalı JSON:")
            print(json_str)
            return None
    else:
        print("\nLLM çıktısında JSON bulunamadı.")
        return None

if __name__ == "__main__":
    file_path = "Fatura_Example_1.pdf"  # veya "fatura.pdf"

    print("--- OCR İşlemi Başlatılıyor ---")
    ocr_text = extract_text_from_pdf(file_path) if is_pdf(file_path) else extract_text_from_image(file_path)

    if not ocr_text.strip():
        print("Hata: OCR işlemi başarısız.")
    else:
        print("--- OCR Metni Çıkarıldı ---")
        print("\n--- LLM ile JSON Çıkarımı Başlatılıyor ---")
        parsed_json = parse_invoice_with_llm(ocr_text)

        print("\n--- JSON Çıkarımı Tamamlandı ---")
        if parsed_json:
            print("\nYapılandırılmış Fatura JSON:\n")
            print(json.dumps(parsed_json, indent=2, ensure_ascii=False))
        else:
            print("\nJSON formatına dönüştürülemedi.")
