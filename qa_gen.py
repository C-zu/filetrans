from autorag.data.qacreation import generate_qa_llama_index, make_single_content_qa
import os
from llama_index.llms.together import TogetherLLM
import pandas as pd

llm_mistral7B = TogetherLLM(
    model="mistralai/Mistral-7B-Instruct-v0.3", api_key="c4c7f717605e8ad6d86f8dbfa79314c592a7cb9a07603c41194882be4ee35ced"
)

llm_llama3 = TogetherLLM(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo", api_key="c4c7f717605e8ad6d86f8dbfa79314c592a7cb9a07603c41194882be4ee35ced"
)

llm_gemma_2 = TogetherLLM(
    model="google/gemma-2-9b-it", api_key="c4c7f717605e8ad6d86f8dbfa79314c592a7cb9a07603c41194882be4ee35ced"
)

llm_databricks = TogetherLLM(
    model="databricks/dbrx-instruct", api_key="c4c7f717605e8ad6d86f8dbfa79314c592a7cb9a07603c41194882be4ee35ced"
)

prompt = """
Bạn là một AI được giao nhiệm vụ chuyển đổi Văn bản về thủ tục dịch vụ công thành bộ câu hỏi TIẾNG VIỆT.

Hướng dẫn:
Câu hỏi phải liên quan đến Văn bản đã cho.
Mỗi Văn bản cung cấp luôn có Tên thủ tục của Văn bản đó, mỗi câu hỏi BẮT BUỘC chứa Tên thủ tục và câu hỏi xoay quanh nó và TUYẾT ĐỐI không chứa bất kỳ cụm từ chung chung như "thủ tục này", "đề cập trong văn bản", "trong văn bản", "trong thông tin đã cung cấp", "thủ tục nào", "trong thủ tục này" v.v.
LƯU Ý không hỏi những câu hỏi như xác định tên thủ tục.
Câu hỏi phải luôn chính xác, có ý nghĩa tránh sai đại từ.
Câu hỏi phải được đa dạng kiểu hỏi, có thể dùng các từ ngữ đồng nghĩa để diễn giải tự nhiên, thân thiện, không máy móc, cứng nhắc.
Kết quả phải luôn có số lượng câu hỏi được cung cấp.

Ví dụ các mẫu câu hỏi đúng:
Mã của thủ tục đăng ký căn cước công dân là gì?
Để đăng ký tạm trú cần những hồ sơ nào?
Căn cứ pháp lý của thủ tục công nhận làng nghề truyền thống?
Nêu thành phần hồ sơ của thủ tục công nhận nguồn giống cây trồng lâm nghiệp?
Quy trình gia hạn giấy phép khai thác nước ngầm bao gồm những bước nào?

Ví dụ các mẫu câu hỏi sai:
Thủ tục nào yêu cầu nộp hồ sơ trực tiếp hoặc qua dịch vụ bưu chính?
Hình thức nộp hồ sơ cho thủ tục này có thể là gì?

Văn bản:

Chi tiết thủ tục hành chính:
Mã thủ tục:
1.000105
Số quyết định:
1560/QĐ-LĐTBXH
Tên thủ tục:
Báo cáo giải trình nhu cầu, thay đổi nhu cầu sử dụng người lao động nước ngoài
Cấp thực hiện:
Cấp Tỉnh
Loại thủ tục:
TTHC được luật giao quy định chi tiết
Lĩnh vực:
Việc làm

Kết quả với 2 câu hỏi
[Q]: Thủ tục báo cáo giải trình nhu cầu, thay đổi nhu cầu sử dụng người lao động nước ngoài thuộc loại thủ tục gì?
[Q]: Thủ tục báo cáo giải trình nhu cầu, thay đổi nhu cầu sử dụng người lao động nước ngoài có mã là gì?

Văn bản:

{{text}}

Kết quả với {{num_questions}} câu hỏi:
"""

prompt2 = prompt = """
Bạn là một AI được giao nhiệm vụ chuyển đổi văn bản về thủ tục dịch vụ công thành bộ câu hỏi và trả lời TIẾNG VIỆT.

Hướng dẫn:
1. Nguồn thông tin*: Cả câu hỏi và câu trả lời phải được trích xuất trực tiếp từ văn bản đã cho. Không suy đoán hoặc thêm thông tin ngoài những gì có trong văn bản.
2. Yêu cầu về câu hỏi:
-Mỗi câu hỏi phải bao gồm Tên thủ tục cụ thể có trong văn bản. Tránh sử dụng các cụm từ chung chung như "thủ tục này", "trong văn bản", "thủ tục nào", v.v.
-Các câu hỏi phải chính xác và liên quan trực tiếp đến thông tin được cung cấp. Tránh hỏi những câu hỏi không rõ ràng hoặc không liên quan.
-Đảm bảo mỗi câu hỏi khác nhau về cấu trúc nhưng vẫn hỏi về những khía cạnh cụ thể liên quan đến thủ tục được đề cập.
3. Yêu cầu về câu trả lời:
-Câu trả lời phải đầy đủ, chính xác và được trích dẫn trực tiếp từ văn bản.
-Trả lời một cách rõ ràng và không mơ hồ, không sử dụng các đại từ chung chung như "nó", "điều này", v.v.
4.Đa dạng câu hỏi: Câu hỏi phải có tính đa dạng về cách diễn đạt nhưng vẫn giữ được sự nhất quán về nội dung thông tin. Dùng các từ đồng nghĩa hoặc cách diễn đạt khác nhau để hỏi về các khía cạnh khác nhau của thủ tục.
5.Số lượng câu hỏi: Kết quả phải bao gồm số lượng câu hỏi và câu trả lời được yêu cầu.

Ví dụ về câu hỏi:
- Mã của thủ tục [Tên thủ tục] là gì?
- Để thực hiện [Tên thủ tục], cần những hồ sơ nào?
- Quy trình thực hiện [Tên thủ tục] bao gồm những bước nào?
- Điều kiện cần thiết để hoàn thành [Tên thủ tục] là gì?
Văn bản:

Chi tiết thủ tục hành chính:
Mã thủ tục:
1.000105
Số quyết định:
1560/QĐ-LĐTBXH
Tên thủ tục:
Báo cáo giải trình nhu cầu, thay đổi nhu cầu sử dụng người lao động nước ngoài
Cấp thực hiện:
Cấp Tỉnh
Loại thủ tục:
TTHC được luật giao quy định chi tiết
Lĩnh vực:
Việc làm

Kết quả với 2 câu hỏi và trả lời:
[Q]: Thủ tục báo cáo giải trình nhu cầu, thay đổi nhu cầu sử dụng người lao động nước ngoài thuộc loại thủ tục gì?
[A]: Thủ tục báo cáo giải trình nhu cầu, thay đổi nhu cầu sử dụng người lao động nước ngoài thuộc loại thủ tục TTHC được luật giao quy định chi tiết.
[Q]: Thủ tục báo cáo giải trình nhu cầu, thay đổi nhu cầu sử dụng người lao động nước ngoài có mã là gì?
[A]: Mã thủ tục của thủ tục báo cáo giải trình nhu cầu, thay đổi nhu cầu sử dụng người lao động nước ngoài là 1.000105.

Văn bản:

{{text}}

Kết quả với {{num_questions}} câu hỏi và trả lời:
"""

prompt3 = """
Bạn là một AI được giao nhiệm vụ chuyển đổi văn bản về thủ tục dịch vụ công thành bộ câu hỏi và trả lời TIẾNG VIỆT.

Hướng dẫn:
1. Nguồn thông tin: Cả câu hỏi và câu trả lời phải được trích xuất trực tiếp từ văn bản đã cho. Không suy đoán hoặc thêm thông tin ngoài những gì có trong văn bản.
2. Yêu cầu về câu hỏi:
-Không hỏi về tên của thủ tục.
-Mỗi câu hỏi phải bao gồm "Tên thủ tục" cụ thể có trong văn bản.
-Không được sử dụng các cụm từ chung chung như "thủ tục này", "trong văn bản", "thủ tục nào".
-Các câu hỏi phải chính xác và liên quan trực tiếp đến thông tin được cung cấp. Tránh hỏi những câu hỏi không rõ ràng hoặc không liên quan.
-Đảm bảo mỗi câu hỏi khác nhau về cấu trúc nhưng vẫn hỏi về những khía cạnh cụ thể liên quan đến thủ tục được đề cập.
3. Yêu cầu về câu trả lời:
-Câu trả lời phải đầy đủ, chính xác và được trích dẫn trực tiếp từ văn bản.
-Trả lời một cách rõ ràng và không mơ hồ, không sử dụng các đại từ chung chung như "nó", "điều này", v.v.
4.Đa dạng câu hỏi: Câu hỏi phải có tính đa dạng về cách diễn đạt nhưng vẫn giữ được sự nhất quán về nội dung thông tin. Dùng các từ đồng nghĩa hoặc cách diễn đạt khác nhau để hỏi về các khía cạnh khác nhau của thủ tục.
5.Số lượng câu hỏi: Kết quả phải bao gồm số lượng câu hỏi và câu trả lời được yêu cầu.

Ví dụ về câu hỏi sai:
Tên của thủ tục này là gì?
Cấp thực hiện của thủ tục này là gì?
Loại thủ tục của thủ tục này là gì ?


Kết quả với 2 câu hỏi và trả lời:
[Q]: Thủ tục báo cáo giải trình nhu cầu, thay đổi nhu cầu sử dụng người lao động nước ngoài thuộc loại thủ tục gì?
[A]: Thủ tục báo cáo giải trình nhu cầu, thay đổi nhu cầu sử dụng người lao động nước ngoài thuộc loại thủ tục TTHC được luật giao quy định chi tiết.
[Q]: Thủ tục báo cáo giải trình nhu cầu, thay đổi nhu cầu sử dụng người lao động nước ngoài có mã là gì?
[A]: Mã thủ tục của thủ tục báo cáo giải trình nhu cầu, thay đổi nhu cầu sử dụng người lao động nước ngoài là 1.000105.

Văn bản:

{{text}}

Kết quả với {{num_questions}} câu hỏi và trả lời:

"""

corpus_df_temp = pd.read_parquet('corpus.parquet')
qa_df = make_single_content_qa(corpus_df_temp, 100, generate_qa_llama_index, llm=llm_databricks, question_num_per_content=5, output_filepath='./data/llm_question_gen/qa_databricks.parquet', cache_batch=4,prompt=prompt3,upsert=True)