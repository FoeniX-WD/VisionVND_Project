import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
from gtts import gTTS
import io
import os

# Bản đồ ánh xạ từ label AI sang text tiếng Việt
MONEY_MAP = {
    "1k": "Một nghìn đồng",
    "2k": "Hai nghìn đồng",
    "5k": "Năm nghìn đồng",
    "10k": "Mười nghìn đồng",
    "20k": "Hai mươi nghìn đồng",
    "50k": "Năm mươi nghìn đồng",
    "100k": "Một trăm nghìn đồng",
    "200k": "Hai trăm nghìn đồng",
    "500k": "Năm trăm nghìn đồng",
    "10000": "Mười nghìn đồng",
    "20000": "Hai mươi nghìn đồng",
    "50000": "Năm mươi nghìn đồng",
    "100000": "Một trăm nghìn đồng",
    "200000": "Hai trăm nghìn đồng",
    "500000": "Năm trăm nghìn đồng"
}

# ======= CẤU HÌNH GIAO DIỆN =======
st.set_page_config(page_title="VisionVND - Trợ Lý Tiền Mặt", page_icon="👁️", layout="wide")

# CSS Cao cấp dành cho Hackathon MVP
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700;800&display=swap');
    
    * {
        font-family: 'Space Grotesk', sans-serif;
    }
    
    /* Ghi đè nền Default thành màu Radial Gradient Sang Trọng */
    div[data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at 50% 0%, #1e293b 0%, #0f172a 50%, #020617 100%);
        color: white;
    }
    div[data-testid="stHeader"] {
        background-color: transparent !important;
    }
    
    /* Title Layout - Premium Glassmorphism */
    .title-container {
        text-align: center;
        padding: 3rem 0 2rem 0;
        background: linear-gradient(180deg, rgba(15,23,42,0.8) 0%, rgba(15,23,42,0) 100%);
        border-radius: 0 0 50% 50% / 0 0 40px 40px;
        margin-top: -60px;
        margin-bottom: 50px;
        backdrop-filter: blur(12px);
        box-shadow: 0 15px 40px rgba(0,0,0,0.3);
    }
    
    .main-title {
        font-size: 4.5rem;
        font-weight: 800;
        background: linear-gradient(to right, #38bdf8, #a78bfa, #38bdf8);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shineTitle 4s linear infinite;
        margin: 0;
        letter-spacing: -1px;
    }

    @keyframes shineTitle {
        to { background-position: 200% center; }
    }
    
    /* Box Kết Quả */
    .result-card {
        background: rgba(30, 41, 59, 0.4);
        border-radius: 30px;
        padding: 50px 30px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        text-align: center;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4), inset 0 0 20px rgba(255, 255, 255, 0.02);
        position: relative;
        overflow: hidden;
        backdrop-filter: blur(20px);
        transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1), box-shadow 0.3s;
        animation: slideUp 0.6s cubic-bezier(0.16, 1, 0.3, 1);
    }
    
    .result-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 30px 60px rgba(0, 0, 0, 0.5), inset 0 0 30px rgba(255, 255, 255, 0.05);
    }
    
    .result-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; height: 6px;
        background: linear-gradient(90deg, #10b981, #3b82f6, #8b5cf6, #10b981);
        background-size: 300% auto;
        animation: gradientMove 3s linear infinite;
    }

    @keyframes gradientMove {
        to { background-position: 300% center; }
    }

    @keyframes slideUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .money-text {
        font-size: 4rem;
        font-weight: 800;
        color: #10b981;
        margin: 20px 0;
        line-height: 1.1;
    }
    
    .scan-status {
        font-size: 1.4rem;
        color: #e2e8f0;
        font-weight: 600;
    }
    
    /* Badge AI Confidence */
    .badge-confidence {
        display: inline-block;
        background: linear-gradient(90deg, rgba(16, 185, 129, 0.15), rgba(59, 130, 246, 0.15));
        color: #34d399;
        padding: 10px 25px;
        border-radius: 30px;
        font-size: 1.2rem;
        font-weight: bold;
        border: 1px solid rgba(52, 211, 153, 0.4);
        box-shadow: 0 0 15px rgba(16, 185, 129, 0.2);
    }

    /* Icon Floating Animation */
    .floating-icon {
        font-size: 6rem;
        display: inline-block;
        opacity: 0.9;
        animation: floatIcon 3s ease-in-out infinite;
        filter: drop-shadow(0 0 20px rgba(255, 255, 255, 0.1));
    }

    @keyframes floatIcon {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-15px); }
    }

    /* Section Titles */
    .section-title {
        color: white;
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 5px;
        border-bottom: 2px solid rgba(255,255,255,0.1);
        padding-bottom: 12px;
        letter-spacing: -0.5px;
    }

    /* Override Camera Thumbs style */
    div[data-testid="stCameraInput"] {
        border-radius: 24px;
        overflow: hidden;
        box-shadow: 0 20px 40px rgba(0,0,0,0.5);
        border: 1px solid rgba(255,255,255,0.1);
    }
</style>
""", unsafe_allow_html=True)

# ======= HEADER =======
st.markdown("""
<div class="title-container">
    <h1 class="main-title">👁️ VisionVND</h1>
    <p style="color: #94a3b8; font-size: 1.3rem; margin-top: 15px; font-weight: 400;">Trợ Lý Nhận Diện Tiền Mặt Hỗ Trợ Người Khiếm Thị</p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_yolo_model():
    # Streamlit sẽ tìm từ vị trí Train trước (vì User đang train local)
    if os.path.exists("runs/detect/train/weights/best.pt"):
        return YOLO("runs/detect/train/weights/best.pt")
    elif os.path.exists("best.pt"):
        return YOLO("best.pt")
    return None

def text_to_speech(text):
    try:
        tts = gTTS(text=text, lang='vi')
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        return audio_buffer
    except Exception as e:
        st.toast("⚠️ Mất kết nối Internet, không thể phát giọng nói!", icon="🔇")
        return None

model = load_yolo_model()

# Bố cục giao diện Layout: Cột trái (Camera) - Cột phải (Cảm biến Kết Qua)
col1, space, col2 = st.columns([1.2, 0.1, 1])

with col1:
    st.markdown("<h3 class='section-title'>📷 Màn hình quét</h3>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8;'>Đưa tờ tiền lên phía trước camera của bạn, hệ thống sẽ tự động quét và phát giác ngay lập tức.</p>", unsafe_allow_html=True)
    camera_image = st.camera_input("Quét Tiền", label_visibility="collapsed")

with col2:
    st.markdown("<h3 class='section-title'>📊 Kết Quả Phân Tích</h3>", unsafe_allow_html=True)
    
    if model is None:
        st.markdown("""
        <div class="result-card" style="border-top-color: #ef4444;">
            <p class="scan-status" style="color: #ef4444;">⚠️ AI Đang Học Việc...</p>
            <p style="color: #cbd5e1; font-size: 1.1rem; line-height: 1.6;">Hệ thống phát hiện lệnh Train đang chạy. Hãy kiên nhẫn đợi máy tạo ra não bộ AI <code>best.pt</code> và ứng dụng sẽ tự động thức tỉnh!</p>
        </div>
        """, unsafe_allow_html=True)
    elif camera_image is None:
        st.markdown("""
        <div class="result-card">
            <div class="floating-icon">💰</div>
            <p class="scan-status" style="margin-top: 15px;">Màn hình đang chờ...</p>
            <p style="color: #94a3b8; margin-top: 5px;">Vui lòng cấp quyền Camera Browser để bắt đầu.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # User vừa chụp, tiến hành quét AI
        image = Image.open(camera_image).convert('RGB')
        img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        with st.spinner("🧠 AI đang bóc tách hình ảnh..."):
            results = model(img_bgr)
            detections = results[0].boxes

        if len(detections) > 0:
            # Thu thập logic tốt nhất
            confidences = detections.conf.cpu().numpy()
            best_idx = int(np.argmax(confidences))
            best_class_idx = int(detections.cls[best_idx].item())
            confidence = confidences[best_idx] * 100
            
            label_raw = str(model.names[best_class_idx]).lower()
            spoken_text = MONEY_MAP.get(label_raw, f"Phát hiện tờ {label_raw}")
            
            # Khối Highlight Thành Công
            st.markdown(f"""
            <div class="result-card">
                <p class="scan-status">Trị giá được xác nhận:</p>
                <div class="money-text">{spoken_text.upper()}</div>
                <div style="margin-top: 25px;">
                    <span class="badge-confidence">Độ chuẩn xác: {confidence:.2f}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Tự động phát Loa
            audio_stream = text_to_speech(spoken_text)
            if audio_stream is not None:
                st.audio(audio_stream, format='audio/mp3', autoplay=True)
            
            # Expandable hiển thị chi tiết Box dành cho Ban Giám Khảo (ẩn đi khỏi view của người khiếm thị)
            st.markdown("<br/>", unsafe_allow_html=True)
            with st.expander("🔍 Dành cho Giám Khảo: Xem AI định vị Bounding Box"):
                annotated_img_bgr = results[0].plot()
                annotated_img_rgb = cv2.cvtColor(annotated_img_bgr, cv2.COLOR_BGR2RGB)
                st.image(annotated_img_rgb, use_container_width=True)
                
        else:
            # Khối Báo Lỗi
            st.markdown("""
            <div class="result-card" style="border-top: 4px solid #f59e0b;">
                <div class="floating-icon" style="opacity: 1;">❓</div>
                <p class="scan-status" style="color: #f59e0b; margin-top: 20px;">Không tìm thấy tiền!</p>
                <p style="color: #cbd5e1; margin-top: 10px;">Vui lòng đưa phần giữa tờ tiền sát lại Camera hơn hoặc tăng cường độ chiếu sáng.</p>
            </div>
            """, unsafe_allow_html=True)
            
            err_audio = text_to_speech("Chưa dò ra tờ tiền, hãy thử lại")
            if err_audio is not None:
                st.audio(err_audio, format='audio/mp3', autoplay=True)
