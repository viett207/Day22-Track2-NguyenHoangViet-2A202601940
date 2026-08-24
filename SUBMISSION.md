# Day 22 Lab - Submission Summary

## Ket qua

- Nhiem vu 1: RAG voi FAISS, LangChain va 50 LangSmith traces da hoan thanh.
- Nhiem vu 2: hai prompt tren Prompt Hub, pull khi chay va A/B routing tat dinh da hoan thanh.
- Nhiem vu 3: 50 QA x 2 prompt, du 4 RAGAS metrics; faithfulness V1 = 1.0000, V2 = 0.9188.
- Nhiem vu 4: PII detector va JSON formatter co demo log da hoan thanh.
- Unit tests: 7/7 passed.

## Bang chung

- `evidence/01_langsmith_traces.png`
- `evidence/02_prompt_hub.png`
- `evidence/02_ab_routing_log.txt`
- `evidence/03_ragas_scores.png`
- `evidence/03_ragas_report.json`
- `evidence/04_pii_demo_log.txt`
- `evidence/04_json_demo_log.txt`

## Viec can lam thu cong truoc khi nop

- Tao GitHub repository public, commit va push ma nguon; khong commit `.env`.
- Dien URL GitHub repository vao cong nop bai.
- Chia se LangSmith project `day22-lab` neu khoa hoc yeu cau public access, sau do nop URL project.
- Kiem tra giao dien LangSmith hien thi tong cong it nhat 100 traces.

## Ghi chu ve thoi gian hoan thanh

Bai hoan thanh cham hon du kien do quota mien phi cua Gemini bi het trong qua
trinh chay. De hoan thanh day du 100 luot RAG va bon chi so danh gia, du an phai
chuyen sang Ollama chay local. RAGAS va HHEM sau do duoc chay tren CPU, trong do
moi phien ban prompt can xu ly 50 cap QA, nen thoi gian danh gia keo dai dang ke.
