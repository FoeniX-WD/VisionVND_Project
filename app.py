import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
from gtts import gTTS
import io
import os
import time
import pygame
import threading

# Khởi tạo bộ trộn âm thanh của Pygame
pygame.mixer.init()

# Bản đồ ánh xạ từ label AI sang text tiếng Việt
MONEY_MAP = {
    "1k": "Một nghìn đồng", "2k": "Hai nghìn đồng", "5k": "Năm nghìn đồng",
    "10k": "Mười nghìn đồng", "20k": "Hai mươi nghìn đồng", "50k": "Năm mươi nghìn đồng",
    "100k": "Một trăm nghìn đồng", "200k": "Hai trăm nghìn đồng", "500k": "Năm trăm nghìn đồng",
    "10000": "Mười nghìn đồng", "20000": "Hai mươi nghìn đồng", "50000": "Năm mươi nghìn đồng",
    "100000": "Một trăm nghìn đồng", "200000": "Hai trăm nghìn đồng", "500000": "Năm trăm nghìn đồng"
}

st.set_page_config(page_title="VisionVND - Trợ Lý Tiền Mặt", page_icon="👁️", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap');
    * { font-family: 'Space Grotesk', sans-serif; }
    div[data-testid="stAppViewContainer"] { background-color: #0f172a; color: white; }
    div[data-testid="stHeader"] { background-color: transparent !important; }
    .title-container { text-align: center; padding: 2rem 0; background: linear-gradient(180deg, rgba(30,41,59,1) 0%, rgba(15,23,42,1) 100%); border-radius: 0 0 40px 40px; margin-top: -60px; margin-bottom: 40px; box-shadow: 0 10px 40px rgba(0,0,0,0.4); }
    .main-title { font-size: 4rem; font-weight: 700; background: linear-gradient(to right, #38bdf8, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0; }
    .result-card { background: rgba(30, 41, 59, 0.7); border-radius: 30px; padding: 50px 30px; border: 1px solid rgba(255, 255, 255, 0.1); text-align: center; box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4); position: relative; overflow: hidden; backdrop-filter: blur(12px); }
    .result-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 6px; background: linear-gradient(90deg, #10b981, #3b82f6); }
    .money-text { font-size: 3.8rem; font-weight: 700; color: #10b981; margin: 20px 0; text-shadow: 0 0 20px rgba(16, 185, 129, 0.4); line-height: 1.1; }
    .scan-status { font-size: 1.4rem; color: #94a3b8; }
    .badge-confidence { display: inline-block; background: rgba(16, 185, 129, 0.2); color: #10b981; padding: 8px 20px; border-radius: 30px; font-size: 1.1rem; font-weight: bold; border: 1px solid rgba(16, 185, 129, 0.3); }
    .progress-bar-container { width: 100%; background-color: #334155; border-radius: 10px; margin-top: 15px; height: 8px; overflow: hidden; }
    .progress-bar-fill { height: 100%; background-color: #3b82f6; transition: width 0.1s ease; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="title-container">
    <h1 class="main-title">👁️ VisionVND</h1>
    <p style="color: #94a3b8; font-size: 1.3rem; margin-top: 15px; font-weight: 400;">Trợ Lý Nhận Diện Tiền Mặt Hỗ Trợ Người Khiếm Thị</p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_yolo_model():
    if os.path.exists("runs/detect/train/weights/best.pt"): return YOLO("runs/detect/train/weights/best.pt")
    elif os.path.exists("best.pt"): return YOLO("best.pt")
    return None

# Hàm chạy loa ngầm ở Backend (Không phụ thuộc vào trình duyệt Web nữa)
def play_audio_background(text):
    try:
        tts = gTTS(text=text, lang='vi')
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        pygame.mixer.music.load(fp)
        pygame.mixer.music.play()
    except Exception as e:
        print("Lỗi Audio:", e)
        try:
            st.toast("🔇 Mất kết nối Internet, không thể phát giọng nói!", icon="⚠️")
        except:
            pass

model = load_yolo_model()

if 'camera_running' not in st.session_state: st.session_state.camera_running = False
if 'last_detected' not in st.session_state: st.session_state.last_detected = ""

col1, space, col2 = st.columns([1.2, 0.1, 1])

with col1:
    st.markdown("<h3 style='color: white;'>📷 Mắt Thần Khung Hình</h3>", unsafe_allow_html=True)
    toggle_cam = st.toggle("🔴 BẬT / TẮT CAMERA QUÉT LIÊN TỤC", value=st.session_state.camera_running)
    st.session_state.camera_running = toggle_cam
    camera_placeholder = st.empty()

with col2:
    st.markdown("<h3 style='color: white;'>📊 Kết Quả Tư Duy</h3>", unsafe_allow_html=True)
    if model is None:
        st.error("⚠️ AI Đang Học Việc... Thiếu file best.pt")
    else:
        result_placeholder = st.empty()
        if not st.session_state.camera_running:
            result_placeholder.markdown("""
            <div class="result-card"><div style="font-size: 5rem; opacity: 0.5;">💤</div>
            <p class="scan-status" style="margin-top: 10px;">Camera đang tắt...</p></div>
            """, unsafe_allow_html=True)
            st.session_state.last_detected = ""

# ======= VÒNG LẶP QUÉT CAMERA CÓ CƠ CHẾ ĐỒNG THUẬN =======
if st.session_state.camera_running and model is not None:
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.toast("⚠️ Vui lòng cấp quyền Camera Browser hoặc cắm WebCam!", icon="⚠️")
            st.session_state.camera_running = False
            time.sleep(0.5)
            st.rerun()
    except Exception as e:
        st.toast("⚠️ Lỗi truy cập Camera!", icon="⚠️")
        st.session_state.camera_running = False
        time.sleep(0.5)
        st.rerun()
        
    REQUIRED_FRAMES = 15  
    current_candidate = "" 
    consecutive_frames = 0 
    
    while cap.isOpened() and st.session_state.camera_running:
        ret, frame = cap.read()
        if not ret: break
            
        results = model(frame, verbose=False)
        detections = results[0].boxes
        
        annotated_frame = results[0].plot()
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        camera_placeholder.image(annotated_frame, channels="RGB", use_container_width=True)
        
        if len(detections) > 0:
            confidences = detections.conf.cpu().numpy()
            best_idx = int(np.argmax(confidences))
            confidence = confidences[best_idx] * 100
            
            # ĐÃ HẠ NGƯỠNG TỪ 60 XUỐNG 40 ĐỂ DỄ NHẬN DIỆN TỜ 1K, 2K HƠN
            if confidence > 40:
                label_raw = str(model.names[int(detections.cls[best_idx].item())]).lower()
                
                if label_raw == current_candidate:
                    consecutive_frames += 1 
                else:
                    current_candidate = label_raw 
                    consecutive_frames = 1
                
                progress_percent = min(int((consecutive_frames / REQUIRED_FRAMES) * 100), 100)
                
                if consecutive_frames < REQUIRED_FRAMES:
                    result_placeholder.markdown(f"""
                    <div class="result-card">
                        <div style="font-size: 3rem; opacity: 0.8;">⏱️</div>
                        <p class="scan-status" style="margin-top: 10px; color: #3b82f6;">Đang phân tích và khóa mục tiêu...</p>
                        <div class="progress-bar-container">
                            <div class="progress-bar-fill" style="width: {progress_percent}%;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                elif consecutive_frames == REQUIRED_FRAMES:
                    spoken_text = MONEY_MAP.get(current_candidate, f"Phát hiện tờ {current_candidate}")
                    
                    result_placeholder.markdown(f"""
                    <div class="result-card">
                        <p class="scan-status">Trị giá được xác nhận:</p>
                        <div class="money-text">{spoken_text.upper()}</div>
                        <div style="margin-top: 25px;"><span class="badge-confidence">Độ chuẩn xác: {confidence:.2f}%</span></div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if current_candidate != st.session_state.last_detected:
                        # GỌI LOA CHẠY ẨN BẰNG LUỒNG (THREAD) RIÊNG ĐỂ KHÔNG BỊ GIẬT CAMERA
                        audio_thread = threading.Thread(target=play_audio_background, args=(spoken_text,), daemon=True)
                        try:
                            from streamlit.runtime.scriptrunner import add_script_run_ctx
                            add_script_run_ctx(audio_thread)
                        except:
                            pass
                        audio_thread.start()
                        st.session_state.last_detected = current_candidate
            else:
                consecutive_frames = 0 
        else:
            consecutive_frames = 0 
            current_candidate = ""
            
            result_placeholder.markdown("""
            <div class="result-card" style="border-top: 4px solid #f59e0b;">
                <div style="font-size: 4rem;">🔍</div>
                <p class="scan-status" style="color: #f59e0b; margin-top: 15px;">Đang tìm kiếm tiền...</p>
                <p style="color: #cbd5e1; margin-top: 10px;">Vui lòng đưa phần giữa tờ tiền sát lại Camera.</p>
            </div>
            """, unsafe_allow_html=True)
            st.session_state.last_detected = ""
            
        time.sleep(0.05)

    cap.release()