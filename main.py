from langchain.prompts import ChatPromptTemplate
import json
import re
import ocr
from ocr import ocr_yap

# OllamaLLM yerine Ollama kullanılması daha güncel olabilir,
# ancak kodunuzda OllamaLLM yazdığı için onu kullanıyoruz.
# Eğer pip install langchain-community yaptıysanız şunu deneyebilirsiniz:
# from langchain_community.llms import Ollama
from langchain_community.llms import Ollama # langchain_ollama yerine langchain_community.llms kullanıldı
# LLMChain yerine doğrudan zincirleme (pipe) kullanıldığı için LLMChain importu gereksiz.

# Hatanın Sebebi:
# LangChain prompt şablonları, metin içindeki tek süslü parantezleri {} değişken olarak yorumlar.
# Sizin prompt'unuzun "human" kısmında örnek JSON yapısını göstermek için kullandığınız
# süslü parantezler ({ ve }) LangChain tarafından değişken olarak algılanmış ve bu değişkenlere
# değer atanmadığı için KeyError fırlatılmış.
# Çözüm: Prompt içinde değişken olmayan ve sadece metin olarak kalması gereken süslü parantezleri
# çiftleyerek ({ -> {{ ve } -> }}) LangChain'e bunların değişken olmadığını belirtmektir.

# Düzeltilmiş Prompt tanımı
invoice_extraction_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Sen bir fatura/fiş analiz aracı olarak çalışıyorsun. Görevin, verilen OCR çıktısından bilgileri doğru ve eksiksiz bir şekilde yapılandırılmış JSON formatında çıkartmak. Her alan zorunludur. Eğer bilgi eksikse, değer boş string (\"\"), null, 0 ya da 0.0 olmalıdır. Sadece JSON çıktısını ver, başka hiçbir açıklama ekleme." # LLM'in sadece JSON vermesi için talimat eklendi
    ),
    (
        "human",
        "OCR Metni:\n{invoice_text}\n\n"
        "Talimatlar:\n"
        "1. Aşağıdaki alanları OCR metninden çıkart:\n"
        "   - Mağaza bilgisi(unvan, adres, fiş numarası, tarih)\n"
        "   - Müşteri bilgisi (isim)\n"
        "   - Ürün kalemleri bir fişte muhtemelen birden fazla ürün kalemi olabilir (ürün adı, kodu (varsa, yoksa boş string), miktar, birim fiyat, satır toplamı)\n" # Ürün kodu açıklaması eklendi
        "   - Ödeme bilgileri (ara toplam, toplam vergi oranı (eğer tek bir oran belirtilmemişse 0.0), toplam vergi tutarı, yuvarlama, genel toplam, ödenen tutar, para üstü)\n\n" # Vergi oranı açıklaması eklendi
        "2. JSON çıktısı aşağıdaki formatta olmalıdır (Değerleri OCR metninden doldur):\n"
        "{{\n"  # Çift süslü parantez
        "  \"magazaBilgisi\": {{\n"  # Çift süslü parantez
        "    \"unvan\": \"\",\n"
        "    \"adres\": \"\",\n"
        "    \"fisNumarasi\": \"\",\n"
        "    \"tarih\": \"\"\n"
        "  }},\n"  # Çift süslü parantez
        "  \"musteriBilgisi\": {{\n"  # Çift süslü parantez
        "    \"isim\": \"\"\n"
        "  }},\n"  # Çift süslü parantez
        "  \"urunKalemleri\": [\n"
        "    {{\n"  # Çift süslü parantez
        "      \"urunAdi\": \"\",\n"
        "      \"urunKodu\": \"\",\n"
        "      \"miktar\": 0,\n"
        "      \"birimFiyat\": 0.0,\n"
        "      \"satirToplam\": 0.0\n"
        "    }}...\n"  # Çift süslü parantez
    
        "  ],\n"
        "  \"odemeBilgileri\": {{\n"  # Çift süslü parantez
        "    \"araToplam\": 0.0,\n"
        "    \"vergiOrani\": 0.0, # Fişte belirtilen KDV oranlarının toplamı değil, genel bir oran varsa o yazılır, yoksa 0.0\n" # Açıklama eklendi
        "    \"vergiTutari\": 0.0, # Toplam KDV Tutarı\n" # Açıklama eklendi
        "    \"yuvarlama\": 0.0,\n"
        "    \"genelToplam\": 0.0,\n"
        "    \"odenenTutar\": 0.0,\n"
        "    \"paraUstu\": 0.0\n"
        "  }}\n"  # Çift süslü parantez
        "}}"  # Çift süslü parantez
    )
])

# Görselden OCR metni al
def extract_text_from_image(image_path):
    # Bu fonksiyonun ocr.py dosyanızda tanımlı olduğunu ve
    # resim yolunu alıp metin döndürdüğünü varsayıyoruz.
    return ocr_yap(image_path)

# OCR'den gelen metni LLM ile yapılandırılmış JSON'a çevir
def parse_invoice_with_llm(text):
    # Ollama'dan LLaMA2 modelini kullanıyoruz (veya seçtiğiniz başka bir model)
    # Ollama servisinin çalıştığından emin olun.
    # Model adını doğru yazdığınızdan emin olun (örn: "llama2", "llama3", "mistral" vb.)
    try:
        llm = Ollama(model="llama2") # OllamaLLM yerine Ollama kullandık
    except Exception as e:
        print(f"Hata: Ollama başlatılamadı. Ollama servisinin çalıştığından emin olun. Detay: {e}")
        return None

    # Zinciri oluştur (Prompt + LLM)
    chain = invoice_extraction_prompt | llm

    try:
        # Zinciri çalıştır ve sonucu al
        result = chain.invoke({"invoice_text": text})
    except Exception as e:
        print(f"Hata: LLM zinciri çalıştırılırken bir sorun oluştu: {e}")
        print("--- Gönderilen Metin Başlangıcı ---")
        print(text[:500] + "..." if len(text) > 500 else text) # Hata ayıklama için metnin başını yazdır
        print("--- Gönderilen Metin Sonu ---")
        return None


    print("\nLLM'den Gelen Ham Cevap:\n")
    print(result) # LLM'in tam çıktısını görmek için

    # LLM çıktısı bazen doğrudan JSON olmaz, metin içinde olabilir.
    # JSON kısmını regex ile ayıklamaya çalışalım.
    # Bu regex, ilk açılan '{' ile son kapanan '}' arasındaki her şeyi alır.
    # Dikkat: Eğer LLM iç içe JSON benzeri yapılar veya metin içinde başka süslü parantezler döndürürse
    # bu regex sorun çıkarabilir. Daha sağlam bir yol, LLM'e sadece JSON döndürmesini sıkıca tembihlemektir.
    json_match = re.search(r'\{.*\}', result, re.DOTALL)

    if json_match:
        json_str = json_match.group(0)
        print("\nRegex ile Ayıklanan JSON String:\n")
        print(json_str)
        try:
            # Ayıklanan string'i JSON olarak parse et
            parsed_result = json.loads(json_str)
            return parsed_result
        except json.JSONDecodeError as e:
            print(f"\nJSONDecodeError: LLM çıktısı geçerli bir JSON değil. Hata: {e}")
            print("Sorunlu JSON String'i:")
            print(json_str)
            return None
    else:
        print("\nUyarı: LLM çıktısında JSON formatına uygun bölüm bulunamadı.")
        print("LLM'in çıktısını kontrol edin, prompt'u veya modeli değiştirmeniz gerekebilir.")
        return None

# Ana akış
if __name__ == "__main__":
    image_path = "fatura2.jpg"  # OCR uygulanacak resmin yolu

    # 1. OCR işlemi
    print("--- OCR İşlemi Başlatılıyor ---")
    ocr_text = extract_text_from_image(image_path)
    print(ocr_text)
    if not ocr_text:
        print("Hata: OCR işlemi metin döndürmedi.")
    else:
        print("--- OCR İşlemi Tamamlandı ---")
        # print("\nOCR Metni:\n") # İsterseniz OCR çıktısını görmek için yorumu kaldırın
        # print(ocr_text)

        # 2. LLM ile JSON parse etme
        print("\n--- LLM ile JSON Çıkarımı Başlatılıyor ---")
        parsed_json = parse_invoice_with_llm(ocr_text)

        print("\n--- LLM İşlemi Tamamlandı ---")

        # 3. Sonucu yazdırma
        if parsed_json:
            print("\nYapılandırılmış Fatura Verisi (JSON):\n")
            # JSON'ı düzgün formatlı ve Türkçe karakterleri koruyarak yazdır
            print(json.dumps(parsed_json, indent=2, ensure_ascii=False))
        else:
            print("\nFatura verisi yapılandırılamadı.")