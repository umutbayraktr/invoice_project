from PIL import Image
import pytesseract

def ocr_yap(image_path, dil='tur'):
  
    try:
        image = Image.open(image_path)
        metin = pytesseract.image_to_string(image, lang=dil)
        return metin
    except Exception as e:
        return f"Hata oluştu: {e}"
