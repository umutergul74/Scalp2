import os
import cv2
import numpy as np
import logging
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import threading
import time
import sqlite3
from inference_sdk import InferenceHTTPClient

# --- YAPILANDIRMA ---
API_SECRET = "kumes-esp32-gizli-anahtar-2026xQ9m"
ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "w7yZMPOPEdSpp8kj82C0")
MAX_KARE_BOYUT = 5 * 1024 * 1024  
DB_YOLU = '/root/kumes_veritabani.db' 

# --- LOGLAMA ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
log = logging.getLogger("kumes")

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": ["https://umutergul.me","http://localhost:5173"]}})

# --- GLOBAL HAFIZA (ASENKRON YAPI İÇİN GÜNCELLENDİ) ---
kumes_verisi = {"eggCount": 0, "temp": 0.0, "hum": 0, "foodLevel": 0}
son_veri_zamani = 0
son_kamera_zamani = 0

# YENİ: Video akışı ve AI kutuları birbirinden ayrıldı
son_ham_kare = None 
son_tahminler = [] 
kare_kilidi = threading.Lock()

CLIENT = InferenceHTTPClient(api_url="https://detect.roboflow.com", api_key=ROBOFLOW_API_KEY)

# --- VERİTABANI BAŞLATMA ---
def veritabani_kurulumu():
    try:
        baglanti = sqlite3.connect(DB_YOLU)
        imlec = baglanti.cursor()
        imlec.execute('''
            CREATE TABLE IF NOT EXISTS sensor_verileri (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tarih DATETIME DEFAULT CURRENT_TIMESTAMP,
                sicaklik REAL,
                nem INTEGER,
                yem_seviyesi INTEGER,
                yumurta_sayisi INTEGER
            )
        ''')
        baglanti.commit()
        baglanti.close()
    except Exception as e:
        log.error("Veritabani kurulum hatasi: %s", str(e))

veritabani_kurulumu() 

def api_key_gerekli(f):
    from functools import wraps
    @wraps(f)
    def kontrol(*args, **kwargs):
        if request.headers.get("X-API-Key") != API_SECRET:
            return jsonify({"error": "Yetkisiz"}), 403
        return f(*args, **kwargs)
    return kontrol

# =========================================================
# --- YENİ: ARKA PLAN YAPAY ZEKA İŞÇİSİ (THREAD) ---
# =========================================================
def ai_arkaplan_iscisi():
    global son_ham_kare, son_tahminler, kumes_verisi
    log.info("Yapay Zeka arka plan isleyicisi basladi.")
    
    while True:
        if son_ham_kare is not None:
            try:
                # O anki kareyi kopyala
                kare_kopya = son_ham_kare 
                nparr = np.frombuffer(kare_kopya, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if img is not None:
                    # Roboflow'a sor (Burası yarım saniye sürebilir, video akışını ASLA etkilemez)
                    result = CLIENT.infer(img, model_id="egg-qg6kr/4")
                    predictions = result.get("predictions", [])
                    
                    gecerli_tahminler = [p for p in predictions if p['confidence'] > 0.80]

                    with kare_kilidi:
                        son_tahminler = gecerli_tahminler
                        kumes_verisi["eggCount"] = len(gecerli_tahminler)
                        
            except Exception as e:
                log.error("AI Thread Hatasi: %s", str(e))
        
        # Saniyede en fazla 2 kere AI taraması yap (Sunucuyu yormamak için)
        time.sleep(0.5) 

# Arka plan işçisini sunucu başlarken çalıştır
threading.Thread(target=ai_arkaplan_iscisi, daemon=True).start()

# --- API ENDPOINTS ---
@app.route('/api/veri_gonder', methods=['POST'])
@api_key_gerekli
def veri_gonder():
    global kumes_verisi, son_veri_zamani
    data = request.get_json()
    if not data: return jsonify({"error": "Veri yok"}), 400

    izinli_alanlar = {"temp", "hum", "foodLevel"}
    for key in data:
        if key in izinli_alanlar:
            kumes_verisi[key] = data[key]
            
    guncel_yumurta = kumes_verisi.get("eggCount", 0)

    try:
        baglanti = sqlite3.connect(DB_YOLU)
        imlec = baglanti.cursor()
        imlec.execute('''
            INSERT INTO sensor_verileri (sicaklik, nem, yem_seviyesi, yumurta_sayisi)
            VALUES (?, ?, ?, ?)
        ''', (kumes_verisi.get("temp", 0), kumes_verisi.get("hum", 0), kumes_verisi.get("foodLevel", 0), guncel_yumurta))
        baglanti.commit()
        baglanti.close()
    except Exception as e:
        pass

    son_veri_zamani = time.time()
    return jsonify({"status": "success"}), 200

@app.route('/api/veriler', methods=['GET'])
def veriler():
    simdi = time.time()
    try:
        baglanti = sqlite3.connect(DB_YOLU)
        imlec = baglanti.cursor()
        imlec.execute('''SELECT sicaklik, nem, yem_seviyesi, yumurta_sayisi FROM sensor_verileri ORDER BY id DESC LIMIT 1''')
        kayit = imlec.fetchone()
        baglanti.close()
        temp, hum, foodLevel, eggCount = kayit if kayit else (0, 0, 0, 0)
    except Exception:
        temp, hum, foodLevel, eggCount = 0, 0, 0, 0

    return jsonify({
        "temp": temp, "hum": hum, "foodLevel": foodLevel, "eggCount": eggCount,
        "espCevrimici": (simdi - son_veri_zamani) < 30,
        "kameraCevrimici": (simdi - son_kamera_zamani) < 10
    }), 200

@app.route('/api/gecmis_veriler', methods=['GET'])
def gecmis_veriler():
    try:
        baglanti = sqlite3.connect(DB_YOLU)
        imlec = baglanti.cursor()
        imlec.execute('''SELECT datetime(tarih, '+3 hours'), sicaklik, nem, yem_seviyesi, yumurta_sayisi FROM sensor_verileri ORDER BY id DESC LIMIT 20''')
        kayitlar = imlec.fetchall()
        baglanti.close()
        
        kayitlar.reverse()
        gecmis_listesi = [{"saat": k[0].split(' ')[1] if ' ' in k[0] else k[0], "Sıcaklık": k[1], "Nem": k[2], "Yem": k[3], "Yumurta": k[4]} for k in kayitlar]
        return jsonify(gecmis_listesi), 200
    except Exception:
        return jsonify([]), 500

@app.route('/api/kamera_yukle', methods=['POST'])
@api_key_gerekli
def kamera_yukle():
    global son_ham_kare, son_kamera_zamani
    # YENİ: ESP32'yi hiç bekletme, kareyi RAM'e at ve anında "Tamam" de.
    son_ham_kare = request.data
    son_kamera_zamani = time.time()
    return "Kare alindi", 200

@app.route('/api/kamera_izle')
def kamera_izle():
    def akis():
        islenen_son_kare_zamani = 0  # YENİ: Hafıza değişkeni
        
        while True:
            # SİHİR BURADA: Sadece "yeni" bir kare geldiyse CPU'yu çalıştır
            if son_ham_kare is not None and son_kamera_zamani > islenen_son_kare_zamani:
                
                nparr = np.frombuffer(son_ham_kare, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if img is not None:
                    with kare_kilidi:
                        tahminler = son_tahminler.copy()
                        
                    for pred in tahminler:
                        x, y, w, h = int(pred['x']), int(pred['y']), int(pred['width']), int(pred['height'])
                        x1, y1 = int(x - w/2), int(y - h/2)
                        cv2.rectangle(img, (x1, y1), (x1+w, y1+h), (0, 255, 0), 2)
                        cv2.putText(img, f"Yumurta %{int(pred['confidence']*100)}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    _, buffer = cv2.imencode('.jpg', img)
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                
                # Bu kareyi işlediğimizi not et
                islenen_son_kare_zamani = son_kamera_zamani
                
            # Eğer yeni kare yoksa mikro saniyede uyu, CPU'yu dinlendir
            time.sleep(0.01) 
            
    return Response(akis(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)