# Tối ưu khung thép 2D – OpenSeesPy + NSGA-II (Streamlit)

## Chạy trên máy
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Đưa lên web miễn phí (Streamlit Community Cloud)
1. Tạo repo GitHub (public), đưa vào `app.py` và `requirements.txt`.
2. Vào https://share.streamlit.io → đăng nhập bằng GitHub → **Create app**.
3. Chọn repo, branch, file `app.py`. Mở **Advanced settings** → chọn **Python 3.11**
   (openseespy ổn định nhất ở bản này) → **Deploy**.
4. Nhận link dạng `https://<ten-app>.streamlit.app` để chia sẻ.

Phương án khác: Hugging Face Spaces (SDK Streamlit), CPU miễn phí mạnh hơn (2 vCPU, 16 GB RAM).

## Lưu ý
- Gói miễn phí chỉ có ~1 CPU: mỗi lần phân tích OpenSees mất khoảng 0,05–0,5 s,
  nên demo để mặc định 40 cá thể × 30 thế hệ. Bộ 100 × 500 như notebook nên chạy trên máy.
- OpenSees dùng trạng thái toàn cục, nên app khóa lại để mỗi lúc chỉ một người chạy phân tích.
