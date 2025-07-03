# Amerikan İşaret Dili Tanıma – Videoya Dayalı Bi-GRU Modeli

Bu proje, Amerikan İşaret Dili (ASL) kelimelerini video kliplerden otomatik olarak tanımayı amaçlamaktadır. Google'ın MediaPipe kütüphanesi ile el ve vücut eklemleri tespit edilmekte ve bu verilerle Bi-GRU yapay zeka modeli eğitilmektedir.

## 📁 Kullanılan Teknolojiler
- Python
- TensorFlow ve Keras
- MediaPipe
- OpenCV
- NumPy, Pandas
- Matplotlib
- Tkinter (Grafik Arayüz)

## 📊 Veri Seti
- ASLLRP veri seti (Rutgers Üniversitesi)
- 51 farklı işaret için 639 video
- Dağılım: %70 Eğitim / %20 Doğrulama / %10 Test
📊 Ek istatistikler ve performans sonuçları için [assets/Statistics.txt](./assets/Statistics.txt) dosyasına göz atabilirsiniz.


## ⚙️ Ön İşleme Aşamaları
- Yüz noktaları ve Z-ekseni çıkarıldı
- Sadece omuz, dirsek, bilek ve iki elin 42 noktası kullanıldı
- Koordinatlar omuzlara göre bağıl hale getirildi
- Eksik veriler stack yöntemi ile tamamlandı

## 🤖 Yapay Zeka Modeli – Bi-GRU
- İki katmanlı Bi-GRU modeli (128 ve 64 hücre)
- Masking, Dropout ve BatchNormalization katmanları
- Kullanılan callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
- Dengesiz veriler için class_weight kullanıldı

## 🔍 Sonuçlar
|          Model           | Doğruluk (Validation) | Doğruluk (Gerçek Videolar) |
|--------------------------|-----------------------|----------------------------|
| RNN (mutlak)             |          %38          |             -              |
| LSTM (mutlak)            |          %41          |             -              |
| GRU (mutlak)             |          %44          |         1/10 doğru         |
| GRU (göreli)             |          %60          |         3/10 doğru         |
| **Bi-GRU (nihai model)** |        **%65.5**      |   **%68.2** (30/44 video)  |

## 📦 Gereksinimler

Gerekli kütüphaneleri yüklemek için:

```bash
pip install -r requirements.txt
```

## 🧪 Çalıştırma (Grafik Arayüz)

1. Grafik arayüzü başlatmak için:

```bash
python sign_language_gui.py
```

2. Arayüzde:

- **Load Video** butonuna tıklayarak bir video dosyası seçin.
- **Classify Video** ile sınıflandırmayı başlatın.
- Sonuç ekranda aşağıda görünecektir.

## 🎬 Örnek Videolar

Bazı gerçek ASL işaret videoları test amacıyla [samples](./samples) klasörüne eklenmiştir.

Arayüz üzerinden denemek için:
1. "Load Video" butonuna tıklayın
2. Herhangi bir örnek videoyu seçin
3. "Classify Video" ile sınıflandırmayı başlatın


## ⚠️ Önemli Notlar

- Amerikan İşaret Dili'nde bazı işaretler çok benzer el hareketlerine sahiptir ve bu durum sınıflandırmada karışıklığa neden olabilir.
- Örneğin:
  - `ANSWER` ve `DIRECT`
  - `BIG` ve `COUCH`
  - `ART` ve `CANCEL`

Model bu işaretlerden birini doğru, diğerini yanlış tahmin ederse, bu durum büyük ihtimalle hareketlerin benzerliğinden kaynaklanmaktadır.



> `GRU_model_rel_best.keras` modeli ve `label_map.json` etiket dosyasının yollarının doğru ayarlandığından emin olun.