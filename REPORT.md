# BÁO CÁO THỰC HÀNH: MEMORY SYSTEMS FOR AI AGENT

* **Họ và tên**: Trần Trung Kiên
* **MSSV**: 2A202600850
* **Môn học**: Phase 2 - Track 3 - Day 17 (Memory Systems)

---

## 1. Tổng Quan Bài Lab
Bài lab tập trung vào việc nghiên cứu, xây dựng và so sánh hai kiến trúc bộ nhớ cho AI Agent:
1. **Baseline Agent**: Bộ nhớ ngắn hạn trong phiên (Within-session memory), quên toàn bộ thông tin khi chuyển sang thread hội thoại mới.
2. **Advanced Agent**: Kết hợp 3 tầng bộ nhớ: bộ nhớ ngắn hạn, bộ nhớ dài hạn bền vững lưu trữ trong file `User.md` (`UserProfileStore`) và bộ nhớ cô đọng (`CompactMemoryManager`) tóm tắt hội thoại cũ khi vượt ngưỡng token.

---

## 2. Kiến Trúc Bộ Nhớ Advanced Agent & Các Kỹ Thuật Đã Áp Dụng

### 2.1. Ước lượng Token tối ưu (`estimate_tokens`)
* Triển khai hàm ước lượng token heuristic dựa trên độ dài chuỗi ký tự theo tỷ lệ $1\text{ token} \approx 4\text{ ký tự}$ sau khi đã loại bỏ khoảng trắng thừa đầu cuối. Giải pháp này giúp tăng tốc độ xử lý offline tối đa và tránh gọi API đo đạc token liên tục.

### 2.2. Trích xuất thực thể bền vững (`UserProfileStore` & `extract_profile_updates`)
* **Trích xuất thông tin tiếng Việt chính xác**: Thiết kế regex kết hợp phát hiện các Danh từ riêng viết hoa tiếng Việt (`[A-ZÀ-ỹ]`) để trích xuất chính xác địa danh (`Huế`, `Đà Nẵng`) và tên riêng (`DũngCT`), loại bỏ các từ thường đi kèm.
* **Bộ lọc câu nghi vấn (Anti-Question Filter)**: Loại bỏ các câu hỏi ngược từ người dùng (như "Bạn có biết DũngCT không?") để tránh lưu trữ facts nhiễu vào file hồ sơ.
* **Thứ tự ưu tiên nghề nghiệp**: Đặt logic lọc nghề nghiệp MLOps engineer trước Backend engineer để nhận diện chính xác các turn đính chính nghề nghiệp.

### 2.3. Quản lý nén bộ nhớ (`CompactMemoryManager` & `summarize_messages`)
* Khi tổng dung lượng tokens của thread vượt ngưỡng cấu hình (`compact_threshold_tokens`), tác nhân sẽ tự động trích xuất các tin nhắn cũ bên ngoài vùng bảo lưu (`compact_keep_messages`) để đưa vào hàm tóm tắt.
* **Chống phình nén đệ quy (Anti-Recursive summary nesting)**: Khi phản hồi offline, Advanced Agent chỉ trả về các facts cấu trúc trong `User.md` chứ không nhúng đè summary cũ vào text phản hồi của assistant. Nhờ đó ngăn chặn được việc summary cũ bị lồng đệ quy qua các đợt nén tiếp theo, giúp số lượng token tiêu thụ không bị tăng trưởng theo hàm mũ.

### 2.4. Tính năng Bonus đạt điểm tối đa (Mốc 90-100 điểm)
* **Conflict handling (Giải quyết xung đột thông tin)**: Khi người dùng đính chính thông tin (ví dụ: chuyển nơi ở từ Huế về Đà Nẵng, nghề nghiệp từ Backend sang MLOps), hệ thống tự động ghi đè giá trị mới nhất lên file `User.md` và loại bỏ hoàn toàn thông tin cũ bị sai lệch.
* **Entity extraction (Trích xuất thực thể cấu trúc)**: Phân tách các thông tin trích xuất được thành các thuộc tính định hình sẵn (Tên, Nơi ở, Nghề nghiệp, Phong cách phản hồi, Đồ ăn, Đồ uống, Thú cưng) giúp cấu trúc của `User.md` luôn mạch lạc, dễ truy vấn.

---

## 3. Bảng Kết Quả Đánh Giá (Benchmark Results)

### 3.1. Standard Benchmark (10 cuộc hội thoại ngắn)
| Agent Name | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Agent (Offline)** | 1,874 | 12,626 | **0.0%** | 15.0% | 0 | 0 |
| **Advanced Agent (Offline)** | 6,206 | 36,370 | **73.8%** | 79.0% | 291 | 0 |

### 3.2. Long-Context Stress Benchmark (Hội thoại siêu dài)
| Agent Name | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Agent (Offline)** | 2,815,564 | 3,191,486 | **0.0%** | 15.0% | 0 | 0 |
| **Advanced Agent (Offline)** | 733 | 8,648 | **83.3%** | 86.7% | 172 | **3** |

---

## 4. Phân Tích Chuyên Sâu & Trade-off của Hệ Thống Bộ Nhớ

1. **Hiệu năng Recall vượt trội xuyên suốt phiên làm việc**:
   * Nhờ có file `User.md` được lưu trữ dài hạn trên đĩa, Advanced Agent đạt điểm recall vượt trội (**73.8%** ở Standard và **83.3%** ở Stress test) khi được hỏi lại ở các thread hội thoại hoàn toàn mới, trong khi Baseline Agent hoàn toàn quên sạch (**0.0%** recall).
2. **Chi phí Overhead ở hội thoại ngắn**:
   * Ở các hội thoại ngắn, Advanced Agent tiêu thụ lượng prompt tokens lớn hơn Baseline Agent (36,370 so với 12,626). Lý do là Advanced Agent luôn phải nạp thông tin từ file `User.md` làm tiền đề ngữ cảnh đầu vào ở mỗi lượt chat. Do đó, đối với tác vụ ngắn hạn, Baseline Agent tối ưu chi phí hơn.
3. **Hiệu quả nén vượt trội ở hội thoại dài**:
   * Ở kịch bản Stress test siêu dài, Baseline Agent do phải mang theo toàn bộ lịch sử hội thoại thô tích lũy qua từng lượt khiến chi phí token context bùng nổ khủng khiếp (**3,191,486 prompt tokens**).
   * Advanced Agent nhờ có `CompactMemoryManager` tự động thực hiện **3 lần nén** giúp cô đọng ngữ cảnh đầu vào cũ, giảm lượng prompt token xuống chỉ còn **8,648 tokens** (tối ưu gấp **369 lần** chi phí vận hành).

---

## 5. Hướng Dẫn Chạy Thực Nghiệm

1. Kích hoạt môi trường ảo:
   ```powershell
   .venv\Scripts\activate.ps1
   ```
2. Chạy toàn bộ unit tests để kiểm tra độ tin cậy của bộ nhớ:
   ```powershell
   pytest src/test_agents.py
   ```
3. Chạy chạy suite benchmark để hiển thị bảng so sánh:
   ```powershell
   python src/benchmark.py
   ```
