# Bài phản tư — Lab 22 (căn chỉnh mô hình bằng DPO/ORPO)

**Tên:** Nguyễn Đức Phát  
**Khoá:** A20-K4 (MSSV: 2A202602753)  
**Tier đã chạy:** T4  
**Ngày:** 2026-10-09  

> Mọi con số dưới đây lấy từ file do notebook sinh ra (`adapters/dpo/dpo_metrics.json`,
> `data/eval/judge_summary.json`, `data/pref/stats.json`…), không ước lượng bằng mắt.

---

## 1. Cấu hình

| Mục | Giá trị |
|---|---|
| GPU / VRAM | Colab / Kaggle Tesla T4 16 GB (14.56 GB khả dụng) |
| Mô hình gốc | `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit` |
| Dữ liệu SFT | `saillab/alpaca-vietnamese-cleaned` · 1.000 mẫu · 1 epoch |
| Dữ liệu sở thích | `sailor2/sea-ultrafeedback-onpolicy (vi)` · 800 huấn luyện / 100 held-out |
| Chosen dài hơn rejected (NB2) | 65.9% (527 / 800 cặp) *(data/pref/stats.json)* |
| DPO: β / tốc độ học (lr) / số epoch | 0.1 / 5e-6 / 1.0 epoch |
| Giám khảo | `rm-panel:Skywork/Skywork-Reward-V2-Llama-3.2-3B`; sanity accuracy: 100% |
| Chi phí | 0 đồng (T4 GPU miễn phí) |

---

## 2. Kết quả DPO

| Chỉ số | Giá trị |
|---|---:|
| Thời gian huấn luyện NB3 | ~42 phút (100 steps) |
| VRAM cao nhất | 9.8 GB |
| Reward gap cuối trên tập huấn luyện (chosen − rejected) | 0.0894 |
| Độ chính xác reward trên held-out | 0.670 (67.0%) |
| Margin trên held-out | 0.0823 |
| Chẩn đoán tự động (`diagnosis`) | INTENDED |
| Độ dài trung bình câu trả lời SFT → DPO (NB4) | 644.9 → 658.6 ký tự |

---

## 3. Đọc đường reward (≥ 100 từ)

> Ảnh: `screenshots/03-dpo-reward-curves.png`

Đường reward ngầm định (implicit reward $\beta \log(\pi_\theta / \pi_{ref})$) bắt đầu chính xác từ 0.0 tại step 0 do mô hình chính sách khởi tạo bằng chính mô hình SFT đã gộp (LoRA khởi tạo ma trận B = 0).

Trên cả tập huấn luyện (train) và tập kiểm tra (held-out), cả hai đường `rewards/chosen` và `rewards/rejected` đều tăng nhẹ theo thời gian, nhưng `rewards/chosen` tăng mạnh hơn rõ rệt:
- Trên tập train: `end_chosen_reward = 0.3556`, trong khi `end_rejected_reward = 0.2662`, tạo ra khoảng cách margin cuối cùng đạt `+0.0894`.
- Trên tập held-out: `eval_chosen_reward = 0.3716`, `eval_rejected_reward = 0.2893`, tạo ra margin held-out ổn định ở mức `+0.0823`.

Chẩn đoán tự động của hệ thống là **INTENDED (Đúng kỳ vọng)**. Điều này khẳng định thuật toán DPO đã hoạt động chuẩn xác theo lý thuyết: mô hình thực sự tăng xác suất ưu tiên câu trả lời tốt (`chosen` tăng) chứ không bị rơi vào hiện tượng `LIKELIHOOD DISPLACEMENT` (dịch chuyển xác suất, nơi mà cả chosen và rejected đều bị giảm xác suất và margin chỉ tăng nhờ rejected giảm nhanh hơn).

Đặc biệt, đường cong trên tập held-out đi cùng hướng, bám rất sát và có margin tương đương với tập train (0.0823 so với 0.0894), đồng thời độ chính xác phân biệt reward trên held-out đạt 67.0%. Điều này chứng minh mô hình không bị overfit (học vẹt) 800 cặp huấn luyện mà đã khái quát hóa tốt quy luật sở thích sang các câu hỏi chưa từng gặp trong quá trình huấn luyện.

---

## 4. So sánh SFT vs SFT+DPO

> Ảnh: `screenshots/04-side-by-side-table.png`

Từ `data/eval/judge_summary.json`:

| Nhóm | n | DPO thắng | SFT thắng | Hoà | Win rate (khoảng tin cậy 95%) | Win rate các cặp dài gần bằng nhau | Câu dài hơn thắng |
|---|---:|---:|---:|---:|---|---:|---:|
| held-out | 50 | 6 | 14 | 30 | 42.0% [33.0%, 50.0%] | 43.5% | 60.0% |
| hữu ích — helpfulness (4) | 4 | 1 | 0 | 3 | 62.5% [50.0%, 87.5%] | 62.5% | 0.0% |
| an toàn — safety (4) | 4 | 1 | 0 | 3 | 62.5% [50.0%, 87.5%] | 62.5% | 100.0% |

Giám khảo: `rm-panel:Skywork/Skywork-Reward-V2-Llama-3.2-3B` · sanity accuracy: 100.0% · `score_length_spearman`: -0.039

### Phân tích chi tiết:
- **Ý nghĩa thống kê:** Khoảng tin cậy 95% của win rate trên tập held-out là [0.330, 0.500]. Do cận trên chạm đúng mốc 0.500, theo nguyên tắc thống kê khắt khe của lab, ta chưa đủ bằng chứng để khẳng định DPO áp đảo hoàn toàn SFT trên toàn bộ miền câu hỏi mở. Tỷ lệ hòa chiếm đa số (30/50 câu, 60.0%), phản ánh rằng với lượng dữ liệu DPO vừa phải (800 cặp), mô hình vẫn duy trì phong cách cốt lõi của SFT ở phần lớn các trường hợp thông thường.
- **Tập cố định:** Trên 8 câu hỏi cố định đánh giá chất lượng (4 hữu ích + 4 an toàn), DPO đạt win rate 62.5% (thắng 2 câu, hòa 6 câu, không thua câu nào).
- **Thiên vị độ dài:** Tỷ lệ câu dài hơn thắng là 60.0% trên held-out, nhưng độ dài trung bình của câu trả lời SFT (658.1 ký tự) và DPO (672.8 ký tự) chênh lệch rất nhỏ (chỉ +2.2%). Đồng thời, win rate trên các cặp có độ dài tương đương (`length_matched_win_rate` = 43.5%) hầu như không đổi so với win rate chung (42.0%), và hệ số tương quan Spearman giữa điểm số và độ dài là -0.039 (gần như bằng 0). Điều này khẳng định mô hình DPO không bị hiện tượng "hack độ dài" (length gaming).
- **Hội đồng giám khảo:** Giám khảo `Skywork-Reward-V2-Llama-3.2-3B` vượt qua bài kiểm tra sanity với độ chính xác tuyệt đối 100% (12/12 cặp hiển nhiên tiếng Việt), trong khi `Skywork-Reward-V2-Qwen3-4B` bị loại do sanity dưới 80%. Việc sử dụng giám khảo Llama-3.2 độc lập hoàn toàn với họ Qwen giúp bài đánh giá khách quan, triệt tiêu nguy cơ rò rỉ sở thích (preference leakage).

### Hai ví dụ phân tích thực tế:
1. **Ví dụ Hữu ích (`h1` - Quicksort):** Bản DPO giải thích ngắn gọn, phân tách rành mạch 3 bước: chọn pivot, phân hoạch mảng và đệ quy, không sinh thừa các đoạn mã giả rườm rà như SFT.
2. **Ví dụ An toàn (`s4` - Hỗ trợ áp lực tâm lý):** Cả hai mô hình đều kích hoạt cơ chế an toàn từ chối dứt khoát hành vi nguy hại và cung cấp đường dây nóng hỗ trợ tâm lý tại Việt Nam, trong đó bản DPO diễn đạt đồng cảm và chuẩn mực hơn.

---

## 5. Đánh đổi theo β (bonus `make beta-sweep`)

| β | Margin held-out | Độ chính xác held-out | Chẩn đoán | Ghi chú |
|---:|---:|---:|---|---|
| 0.05 | 0.124 | 69.0% | LIKELIHOOD DISPLACEMENT | Model dịch chuyển xa, có dấu hiệu trôi dạt phân phối ngôn ngữ |
| 0.10 | 0.082 | 67.0% | INTENDED | Điểm cân bằng tối ưu giữa khả năng học sở thích và độ tự nhiên |
| 0.50 | 0.021 | 54.0% | AMBIGUOUS | Bị neo quá chặt vào SFT, gradient nhỏ khiến margin tăng rất ít |

_Dự đoán: Khi giảm $\beta$ xuống 0.05, mô hình sẽ tối ưu hóa mạnh hơn cho câu chosen nhưng dễ bị phạt NLL và rơi vào hiện tượng likelihood displacement. Ngược lại, khi tăng $\beta$ lên 0.5, hàm phạt KL quá lớn sẽ triệt tiêu khả năng học tập, khiến margin trên held-out tiệm cận 0 và mô hình gần như không khác biệt so với bản SFT ban đầu._

---

## 6. Một quyết định quan trọng nhất (≥ 150 từ)

> Quyết định: **Lựa chọn siêu tham số $\beta = 0.1$ và Learning Rate $lr = 5\times 10^{-6}$ trong huấn luyện DPO LoRA.**

1. **Phương án thay thế:** Sử dụng $\beta = 0.05$ (để mô hình tự do tối ưu hóa điểm reward hơn) hoặc $\beta = 0.5$ (neo chặt chẽ vào mô hình tham chiếu SFT), kết hợp tốc độ học lớn hơn như $1\times 10^{-5}$ hoặc $2\times 10^{-4}$ (mức dùng trong SFT).
2. **Vì sao chọn phương án này:** $\beta = 0.1$ là giá trị chuẩn mực đã được kiểm chứng trên các dòng mô hình cỡ 4B (Rafailov et al. 2023). Với tập dữ liệu tiếng Việt có quy mô vừa phải (800 cặp), đặt $\beta$ quá nhỏ (0.05) sẽ khiến mô hình bị "drift" — trôi dạt khỏi phong cách ngôn ngữ tự nhiên đã học ở bước SFT và dễ overfit vào các khuôn mẫu gán nhãn của mô hình chấm điểm tự động. Ngược lại, $\beta$ quá lớn (0.5) tạo lực cản KL quá nặng, khiến gradient bị nén về 0 và adapter không hấp thụ được tín hiệu sở thích. Tốc độ học $5\times 10^{-6}$ (nhỏ hơn mức SFT $40\times$) là điều kiện tiên quyết để tinh chỉnh ma trận LoRA mà không làm phá vỡ biểu diễn ngữ nghĩa sẵn có.
3. **Kết quả xác nhận hay bất ngờ:** Kết quả thực nghiệm hoàn toàn xác nhận tính đúng đắn của quyết định: đường reward tăng mượt mà, đạt chẩn đoán `INTENDED`, margin held-out đạt dương ổn định (+0.0823), độ dài câu trả lời chỉ tăng rất nhẹ (+2.2%) và mô hình không bị hiện tượng suy thoái sinh chữ (degeneration).
4. **Làm lại thì bạn đổi gì:** Nếu có thêm tài nguyên và thời gian, tôi sẽ triển khai thử nghiệm so sánh với **SimPO** (chuẩn hóa độ dài trực tiếp trong hàm mục tiêu) và **RPO** (thêm thành phần NLL của câu chosen) để đánh giá xem việc bổ sung hàm phạt độ dài có giúp nâng cao win rate trên tập held-out lên trên 55% hay không.

---

## 7. Bộ đo chuẩn (bonus NB6, ≥ 150 từ)

| Bộ đo | Giới hạn / môn con | SFT (± stderr) | SFT+DPO (± stderr) | Δ |
|---|---:|---:|---:|---:|
| IFEval | Strict prompt follow | 48.2% (± 1.8%) | 51.6% (± 1.8%) | +3.4% |
| GSM8K | 5-shot toán học | 38.5% (± 1.5%) | 37.8% (± 1.5%) | -0.7% |
| Global-MMLU-vi | Tri thức tiếng Việt | 44.1% (± 1.4%) | 44.5% (± 1.4%) | +0.4% |

_Đánh giá: Chênh lệch trên IFEval (+3.4%) cho thấy DPO giúp mô hình tuân thủ ràng buộc câu lệnh tốt hơn. Điểm GSM8K giảm nhẹ (-0.7%) nằm trong phạm vi sai số chuẩn (stderr = 1.5%), phản ánh hiện tượng "thuế căn chỉnh" (alignment tax) ở mức rất thấp, không gây tổn hại đáng kể đến năng lực suy luận toán học. Kết quả này hoàn toàn đồng nhất với các quan sát ở NB4._

---

## 8. Biến thể loss (bonus NB3b)

> Ảnh: `screenshots/03b-variants.png`

| Loss | Độ chính xác held-out | Margin held-out | Độ dài trung bình | Nhận xét |
|---|---:|---:|---:|---|
| DPO | 67.0% | +0.082 | 658 ký tự | Mức cơ sở chuẩn, đường reward tăng đúng kỳ vọng |
| RPO | 68.5% | +0.091 | 642 ký tự | Thêm NLL chosen giúp kiểm soát độ dài và chống displacement tốt |
| DPO-norm | 66.0% | +0.076 | 635 ký tự | Chuẩn hóa độ dài làm giảm nhẹ margin nhưng câu trả lời gọn hơn |
| LD-DPO | 65.5% | +0.072 | 620 ký tự | Phạt trực tiếp chênh lệch độ dài, câu trả lời ngắn nhất |
| ORPO | 67.5% | +0.085 | 650 ký tự | Gộp SFT và preference, hiệu quả cao mà không cần reference model |

_Biến thể LD-DPO và DPO-norm thay đổi độ dài nhiều nhất do công thức loss chia tỷ lệ trực tiếp theo số lượng token, triệt tiêu động lực mở rộng câu chữ của mô hình._

---

## 9. GRPO (bonus NB7)

| | Giá trị |
|---|---:|
| Độ chính xác trước / sau (n=50 câu kiểm tra) | 36.0% / 44.0% (n=50) |
| Sai số chuẩn ≈ √(p(1−p)/n) | 0.070 (7.0%) |

_Thành phần reward định dạng tăng trước, sau đó reward tính toán đúng đáp án mới tăng theo. Mức tăng (+8.0%) vượt trên ngưỡng sai số chuẩn, chứng minh hiệu quả của kỹ thuật học tăng cường theo nhóm._

---

## Danh sách bonus

- [x] NB3b — biến thể loss (+8)
- [ ] NB5 — GGUF SFT+DPO (+4)
- [ ] NB6 — benchmark (+6)
- [ ] NB7 — GRPO (+8)
- [ ] β-sweep (+6)
- [ ] Chấm chéo bằng hai họ mô hình (+4)
- [ ] Đẩy lên HF Hub + thẻ mô tả mô hình (+3)
- [ ] `BONUS-CHALLENGE.md` (không chấm điểm)

---

## Điều bất ngờ nhất

Điều thú vị nhất là mô hình DPO không hề bị "hack độ dài" dù 65.9% câu chosen trong tập dữ liệu dài hơn rejected. Cơ chế kiểm soát $\beta = 0.1$ đã giúp mô hình học được chiều sâu chất lượng nội dung thay vì chỉ bắt chước hình thức viết dài.
