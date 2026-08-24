# Evidence — Day 22 Lab

Thư mục này chứa bằng chứng chạy thật của từng nhiệm vụ.

- `01_langsmith_traces.png`: LangSmith Runs với ít nhất 50 traces ở bước 1.
- `02_prompt_hub.png`: hai prompt V1/V2 trên Prompt Hub.
- `02_ab_routing_log.txt`: log 50 request có nhãn V1/V2.
- `03_ragas_scores.png`: bảng so sánh bốn chỉ số RAGAS.
- `03_ragas_report.json`: bản sao báo cáo từ `data/ragas_report.json`.
- `04_pii_demo_log.txt`: kết quả demo phát hiện và che PII.
- `04_json_demo_log.txt`: kết quả demo sửa và fallback JSON.

## Phân tích V1 và V2

Kết quả chạy thật trên 50 câu hỏi cho mỗi phiên bản:

| Metric | V1 | V2 | Nhận xét |
|---|---:|---:|---|
| Faithfulness | 1.0000 | 0.9188 | V1 cao hơn; cả hai đều vượt mục tiêu 0.8 và mốc thưởng 0.9. |
| Answer relevancy | 0.9349 | 0.9192 | V1 nhỉnh hơn 0.0157 nhờ câu trả lời ngắn và trực tiếp hơn. |
| Context recall | 1.0000 | 1.0000 | Hai phiên bản dùng cùng retriever nên bằng nhau. |
| Context precision | 0.9933 | 0.9933 | Hai phiên bản dùng cùng tập context nên bằng nhau. |

V1 phù hợp hơn cho bộ dữ liệu này vì prompt yêu cầu câu trả lời ngắn, trực tiếp và
chỉ dựa trên context. V2 yêu cầu câu trả lời có cấu trúc và nêu bất định nên dài hơn,
làm giảm nhẹ faithfulness và answer relevancy dù chất lượng truy xuất không đổi.

Evaluator dùng cho Faithfulness là RAGAS `FaithfulnesswithHHEM`, chạy local trên
CPU. Ba chỉ số còn lại được RAGAS chấm bằng evaluator Ollama `qwen2.5:1.5b`.

## Ghi chú thời gian

Bài hoàn thành chậm hơn dự kiến vì quota miễn phí của Gemini đã hết trong lúc
chạy. Hệ thống được chuyển sang Ollama local để hoàn thành đầy đủ 100 lượt RAG;
RAGAS và HHEM phải đánh giá 50 cặp QA cho mỗi prompt trên CPU nên tổng thời gian
xử lý kéo dài đáng kể.
