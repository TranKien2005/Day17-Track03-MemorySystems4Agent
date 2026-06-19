# BÁO CÁO THỰC HÀNH: MEMORY SYSTEMS FOR AI AGENT

* **Họ và tên**: Trần Trung Kiên
* **MSSV**: 2A202600850
* **Môn học**: Phase 2 - Track 3 - Day 17 (Memory Systems)
* **Model sử dụng**: `gemma3:4b` qua Ollama Cloud API

---

## 1. Tổng Quan Bài Lab
Bài lab tập trung vào việc nghiên cứu, xây dựng và so sánh hai kiến trúc bộ nhớ cho AI Agent:
1. **Baseline Agent**: Bộ nhớ ngắn hạn trong phiên (Within-session memory), quên toàn bộ thông tin khi chuyển sang thread hội thoại mới.
2. **Advanced Agent**: Kết hợp 3 tầng bộ nhớ: bộ nhớ ngắn hạn, bộ nhớ dài hạn bền vững lưu trữ trong file `User.md` (`UserProfileStore`) và bộ nhớ cô đọng (`CompactMemoryManager`) tóm tắt hội thoại cũ khi vượt ngưỡng token.

---

## 2. Kiến Trúc Bộ Nhớ Advanced Agent & Các Kỹ Thuật Đã Áp Dụng

### 2.1. Ước lượng Token tối ưu (`estimate_tokens`)
* Triển khai hàm ước lượng token heuristic dựa trên độ dài chuỗi ký tự theo tỷ lệ $1\text{ token} \approx 4\text{ ký tự}$ sau khi đã loại bỏ khoảng trắng thừa đầu cuối. Giải pháp này giúp tăng tốc độ xử lý và tránh gọi API đo đạc token liên tục.

### 2.2. Trích xuất thực thể bền vững (`UserProfileStore` & `extract_profile_updates`)
* **Trích xuất thông tin đa ngôn ngữ**: Thiết kế regex hỗ trợ cả tiếng Việt (`tên là`, `đang ở`) và tiếng Anh (`my name is`, `I live in`), phát hiện Danh từ riêng viết hoa để trích xuất chính xác địa danh và tên riêng.
* **Bộ lọc câu nghi vấn (Anti-Question Filter)**: Loại bỏ các câu hỏi ngược từ người dùng để tránh lưu trữ facts nhiễu vào file hồ sơ.
* **Conflict handling**: Khi người dùng đính chính thông tin (chuyển nơi ở, nghề nghiệp), hệ thống tự động ghi đè giá trị mới nhất lên `User.md` và loại bỏ thông tin cũ.

### 2.3. Quản lý nén bộ nhớ (`CompactMemoryManager` & `summarize_messages`)
* Khi tổng dung lượng tokens của thread vượt ngưỡng cấu hình (`compact_threshold_tokens`), tác nhân tự động trích xuất các tin nhắn cũ bên ngoài vùng bảo lưu (`compact_keep_messages`) vào hàm tóm tắt.
* **Chống phình nén đệ quy**: Advanced Agent chỉ trả về các facts cấu trúc trong `User.md` thay vì nhúng đè summary cũ vào text phản hồi, ngăn token tăng trưởng theo hàm mũ.

### 2.4. Bonus – Tính năng mở rộng (Mốc 90–100 điểm)

#### (a) Entity extraction
Facts được tách thành 7 trường có cấu trúc: Tên, Nơi ở, Nghề nghiệp, Phong cách trả lời, Đồ uống yêu thích, Món ăn yêu thích, Pet. User.md luôn mạch lạc và dễ truy vấn.

#### (b) Conflict handling
Khi user đính chính thông tin (chuyển nơi ở, nghề nghiệp), `merge_updates()` tự động ghi đè giá trị mới nhất và loại bỏ hoàn toàn fact cũ.

#### (c) Confidence threshold
Hàm `extract_profile_updates_with_confidence()` trả về `{key: (value, confidence)}`. Mỗi pattern có điểm tin cậy khác nhau:
- Khai báo tường minh (`my name is X`, `tên là X`) → **0.95**
- Khai báo ngầm (`I am X`, `I'm X`) → **0.75**
- Vị trí mạnh (`live in`, `đang ở`) → **0.90**
- Vị trí yếu (`from`) → **0.70**
- Nghề nghiệp / sở thích (keyword match) → **0.90–0.95**

`UserProfileStore.merge_updates()` chỉ ghi vào `User.md` khi `confidence >= threshold` (mặc định 0.70). Facts dưới ngưỡng bị loại trước khi chạm đến disk, giảm rủi ro lưu sai thông tin nhiễu.

#### (d) Memory decay
Mỗi fact có metadata lưu trong sidecar `{user_id}_meta.json`:
- `mention_count`: số lần fact được xác nhận/cập nhật
- `last_turn`: turn index lần cuối nhắc đến

Điểm decay được tính theo hàm mũ:

$$score = mention\_count \times e^{-\frac{\ln 2}{halflife} \times (current\_turn - last\_turn)}$$

- Fact mới tạo với `mention_count=1` bắt đầu ở score **1.0**
- Sau `halflife` turns không nhắc → score giảm còn **0.5**
- Sau `2 × halflife` turns → score giảm còn **0.25** (stale)

`get_context_for_prompt()` sắp xếp facts theo decay score giảm dần; facts stale (score < 0.25) bị đẩy xuống section riêng với nhãn `[stale, score=X.XX]` thay vì bị xóa hẳn — bảo toàn recall nhưng báo hiệu cho LLM biết thông tin có thể cũ.

**Trade-off**: Decay tạo thêm rủi ro nếu halflife quá ngắn — fact đúng có thể bị đánh dấu stale nếu user không nhắc lại lâu. Cần tune `decay_halflife` phù hợp với usecase (session ngắn vs dài hạn).

---

## 3. Bảng Kết Quả Đánh Giá – Benchmark Thật (gemma3:4b, Ollama Cloud)

### 3.1. Standard Benchmark (10 cuộc hội thoại ngắn)
| Agent Name | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Agent** | 1,874 | 12,626 | **0.0%** | 15.0% | 0 | 0 |
| **Advanced Agent** | 6,885 | 24,985 | **58.1%** | 66.5% | 329 | 135 |

### 3.2. Long-Context Stress Benchmark (Hội thoại siêu dài)
| Agent Name | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Agent** | 2,815,564 | 3,191,486 | **0.0%** | 15.0% | 0 | 0 |
| **Advanced Agent** | 828 | 6,558 | **58.3%** | 66.7% | 185 | **28** |

---

## 4. Phân Tích Chuyên Sâu & Trade-off của Hệ Thống Bộ Nhớ

1. **Hiệu năng Recall vượt trội xuyên suốt phiên làm việc**:
   * Nhờ có file `User.md` được lưu trữ dài hạn trên đĩa, Advanced Agent đạt recall **58.1%** (Standard) và **58.3%** (Stress), trong khi Baseline Agent hoàn toàn quên sạch (**0.0%** recall) vì không có cơ chế lưu trữ xuyên thread.

2. **Chi phí Overhead ở hội thoại ngắn**:
   * Ở các hội thoại ngắn, Advanced Agent tiêu thụ prompt tokens nhiều hơn Baseline (24,985 so với 12,626) vì luôn phải nạp nội dung `User.md` và summary làm tiền đề ngữ cảnh đầu vào mỗi lượt chat.

3. **Hiệu quả nén vượt trội ở hội thoại dài (Kết quả nổi bật nhất)**:
   * Ở Stress test, Baseline Agent mang theo toàn bộ lịch sử hội thoại thô, khiến chi phí token bùng nổ lên **3,191,486 prompt tokens**.
   * Advanced Agent nhờ `CompactMemoryManager` thực hiện **28 lần nén** giảm xuống chỉ còn **6,558 tokens** — tối ưu gấp **~487 lần** chi phí vận hành.

4. **Unit Tests – 4/4 PASSED (gọi LLM thật)**:
   * `test_user_markdown_read_write_edit`: CRUD file `User.md` ✅
   * `test_compact_trigger`: Logic compaction kích hoạt đúng ngưỡng ✅
   * `test_cross_session_recall`: Advanced nhớ Alice + Paris qua thread mới ✅
   * `test_compact_reduces_prompt_load`: Advanced 94 tokens < Baseline 162 tokens ✅

---

## 5. Hướng Dẫn Chạy Thực Nghiệm

1. Kích hoạt môi trường ảo:
   ```powershell
   .venv\Scripts\activate.ps1
   ```
2. Chạy toàn bộ unit tests (gọi LLM thật qua Ollama Cloud):
   ```powershell
   pytest src/test_agents.py -v -s
   ```
3. Chạy suite benchmark đầy đủ:
   ```powershell
   python src/benchmark.py
   ```
