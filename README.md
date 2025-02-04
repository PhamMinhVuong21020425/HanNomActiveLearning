# Han Nom Recognition - Active Learning

## 1. Giới thiệu

Dự án này triển khai các chiến lược Active Learning cho bài toán nhận diện chữ Hán Nôm. Bằng cách sử dụng các phương pháp lấy mẫu chủ động khác nhau, kỹ thuật Active Learning giúp tối ưu hóa hiệu suất mô hình với lượng dữ liệu cần gán nhãn là ít hơn.

## 2. Tổng quan Active Learning

**Active Learning** là một phương pháp giúp **định nghĩa các mẫu dữ liệu cần thiết cho quá trình gán nhãn** (human annotation), đặc biệt hữu ích khi việc thu thập hoặc dán nhãn dữ liệu tốn kém hoặc khó khăn.

Thay vì sử dụng một lượng lớn dữ liệu có nhãn ngay từ đầu, Active Learning giúp mô hình học hiệu quả bằng cách **chọn lọc những dữ liệu quan trọng mà mô hình muốn học để con người gán nhãn**, từ đó cải thiện hiệu suất của mô hình với ít dữ liệu hơn.

### Các thuật ngữ:

-   **Query Strategy:** Thuật toán để chọn mẫu dữ liệu gán nhãn gọi là **_query strategy_** (chiến lược truy vấn) hoặc **_sampling strategy_** (chiến lược lấy mẫu).

-   **Oracle:** Tổ chức chịu trách nhiệm gán nhãn cho dữ liệu (ground truth). Có thể là một outside system hoặc một human annotator.

-   **Stopping Conditions:** Điều kiện để kết thúc quá trình training là khi hiệu suất của mô hình có thể đạt đến mức ổn định, độ chính xác đạt được như kì vọng hoặc quá trình lặp của active learning đạt đến số lượng mẫu dữ liệu được gắn nhãn định trước.

### Mục tiêu của Active Learning:

-   Định nghĩa và đo lường **mức độ quan trọng của từng mẫu dữ liệu** đối với mô hình.
-   Giúp mô hình đạt được độ chính xác mong muốn với **chi phí thấp nhất** (thời gian, công sức và tiền bạc).
-   Kết hợp giữa human và machine intelligence để **tối đa hoá độ chính xác** cho mô hình học máy.
-   Giúp các mô hình học máy đạt được sự **tổng quát hoá** hơn với các loại dữ liệu trên thực tế.

### Active Learning Pipeline:

![Active Learning Pipeline](images/ALPipeline.png)

1. Unlabelled Data (Dữ liệu chưa gán nhãn):

    - Tập dữ liệu đầu vào chưa có nhãn.

    - Mô hình cần xác định những mẫu dữ liệu nào có độ không chắc chắn cao để yêu cầu gán nhãn.

2. Sampling Data (Chọn mẫu dữ liệu):

    - Sử dụng các chiến lược Active Learning như Least Confidence, Entropy Sampling, BALD,... để chọn ra các mẫu dữ liệu có tiềm năng giúp cải thiện mô hình.

3. Human Annotator (Tổ chức gán nhãn):

    - Gán nhãn cho các mẫu dữ liệu được chọn.
    - Sau khi gán nhãn, dữ liệu này được thêm vào tập huấn luyện.

4. Labelled Data (Dữ liệu đã gán nhãn):

    - Tập dữ liệu đã được gán nhãn để huấn luyện mô hình.

5. Training a Model (Huấn luyện mô hình):

    - Mô hình được huấn luyện với tập dữ liệu mới để cải thiện hiệu suất.

6. Making Predictions (Dự đoán dữ liệu mới):

    - Mô hình đưa ra dự đoán trên dữ liệu chưa gán nhãn và quay lại bước chọn mẫu dữ liệu.

    - Quá trình lặp lại cho đến khi đạt được hiệu suất mong muốn.

### Ba hướng data sampling chính:

-   **Random sampling:** là chiến lược đơn giản nhất nhưng cũng thường kém hiệu quả nhất. Từ tập dữ liệu không có nhãn, chọn ngẫu nhiên một tập con các mẫu dữ liệu để gán nhãn và đưa vào training.

-   **Uncertainty sampling:** là một tập hợp các kĩ thuật để đánh giá và lựa chọn các **unlabeled data** dựa trên **mức độ quan trọng** của nó với mô hình. Ví dụ với mô hình **phân loại nhị phân** thì các mẫu dữ liệu cho ngưỡng xác suất gần 0.5 (eg: [0.45 → 0.55]) gọi là **uncertain** hay **confused**. Các mẫu dữ liệu này cần gán nhãn bởi con người.

-   **Diversity sampling:** xác định các mẫu **unlabeled data** mà chúng đang không nằm trong phân phối dữ liệu (**underrepresented**) hay **chưa được biết đến bởi mô hình** ở thời điểm hiện tại. Phương pháp này tập trung tìm các **đặc trưng hiếm gặp** trong dữ liệu huấn luyện hiện tại để tiến hành gán nhãn thêm. Mục tiêu là tìm ra một tập hợp các mẫu có thể đại diện tốt cho toàn bộ phân phối dữ liệu.

## 3. Uncertainty Sampling

> [!NOTE]

> $\phi(x)$ là độ không chắc chắn của mô hình đối với mẫu dữ liệu $x$.

Dự án này triển khai các phương pháp lấy mẫu sau:

### a. Random Sampling

Chọn ra ngẫu nhiên các mẫu dữ liệu để gán nhãn.

### b. Least Confidence Sampling

Chọn ra các mẫu dữ liệu mà hiện tại mô hình có độ tự tin (confidence) khi dự đoán thấp nhất.

$\phi_{LC}(x) = 1 - P_\theta(y_1^*|x)$

Chỉ số này **càng cao** thì độ tự tin khi dự đoán của mô hình cho mẫu dữ liệu đó càng thấp.

### c. Margin Sampling

Chọn mẫu có chênh lệch nhỏ nhất giữa hai lớp dự đoán có xác suất cao nhất.

$\phi_{MC}(x) = P_\theta(y_1^*|x) - P_\theta(y_2^*|x)$

Nếu như khoảng cách này **càng nhỏ** thì chứng tỏ mô hình càng phân vân với mẫu dữ liệu này và cần phải đưa nó vào gán nhãn.

### d. Ratio Sampling

Chọn mẫu dựa trên tỉ lệ giữa hai lớp dự đoán có xác suất cao nhất.

$\phi_{RC}(x) = \frac{P_\theta(y_1^*|x)}{ P_\theta(y_2^*|x)}$

Tỉ lệ này **càng nhỏ** chứng tỏ mô hình càng phân vân với các mẫu dữ liệu này.

### e. Entropy-based Sampling

Chọn mẫu có độ bất định cao nhất dựa trên entropy.

$\phi_{ENT}(x) = -\sum_{i} P_\theta(y_i|x) log_2 P_\theta(y_i|x)$

Entropy là đại lượng để đo mức độ hỗn loạn hay mức độ không ổn định của một phân phối dữ liệu. Chúng ta sẽ lựa chọn các prediction có entropy **cao nhất** để đưa vào gán nhãn.

### f. Monte Carlo Dropout Sampling

**Dropout** là **tắt ngẫu nhiên** một số nơ-ron trong các hidden layers thường sử dụng khi training.

**Monte Carlo Dropout** (MC Dropout) là kĩ thuật sử dụng **dropout trong quá trình inference** để xác định mức độ **uncertainty** của các input samples.

Để thực hiện MC Dropout, mỗi mẫu dữ liệu sẽ được **inference lặp lại N lần** với các dropout khác nhau của mạng neural gốc, sau đó lấy trung bình kết quả của N lần inference.

### g. BALD (Bayesian Active Learning by Disagreement) Sampling

Chọn mẫu dữ liệu dựa trên **thông tin tương hỗ** (mutual information) giữa model output và model parameters (weights/posterior).

$I(y; \omega | x, D_{\text{train}}) \approx - \sum_c \left( \frac{1}{T} \sum_t p_c^t \right) \log \left( \frac{1}{T} \sum_t p_c^t \right) + \frac{1}{T} \sum_{t,c} p_c^t \log p_c^t$

Trong đó:

$p_c^t$ là xác suất dự đoán ra class c của mẫu dữ liệu input $x$ với model parameters $ω_t ∼ \text{posterior } q^∗_θ (ω)$.

$T$ là số lần inference.

Để có được thành phần đầu tiên, ta thực hiện chạy inference nhiều lần, lấy **trung bình các outputs** và đo lường **entropy** trên đó.

Để có được thành phần thứ hai, ta thực hiện chạy inference nhiều lần, tính toán **entropy của output mỗi lần chạy** sau đó tính **giá trị trung bình.**

## 4. Thực nghiệm



### Prerequisites

-   python >= 3.8.0
-   numpy
-   torch
-   torchvision
-   pillow
-   matplotlib
-   tqdm
-   wandb
