import easyocr

class OCRProcessor:
    def __init__(self, languages=None):
        if languages is None:
            languages = ['en']
        self.reader = easyocr.Reader(languages)

    def extract_text(self, image_path):
        results = self.reader.readtext(image_path)
        extracted_text = ' '.join([text[1] for text in results])
        return extracted_text
    

if __name__ == "__main__":
    ocr = OCRProcessor(languages=['en', 'it'])
    image_path = 'modules\\video\\frames\\frame_at_1641s.jpg'
    text = ocr.extract_text(image_path)
    print("Extracted Text:", text)