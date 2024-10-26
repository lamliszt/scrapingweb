from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
import time
from bs4 import BeautifulSoup

# Tạo template prompt để phân tích
template = (
    "You are tasked with extracting specific information from the following text content: {dom_content}. "
    "Please follow these instructions carefully: \n\n"
    "1. **Extract Information:** Only extract the information that directly matches the provided description: {parse_description}. "
    "2. **No Extra Content:** Do not include any additional text, comments, or explanations in your response. "
    "3. **Empty Response:** If no information matches the description, return an empty string ('')."
    "4. **Direct Data Only:** Your output should contain only the data that is explicitly requested, with no other text."
)

# Khởi tạo mô hình Ollama
model = OllamaLLM(model="llama3.1")

# Hàm quét nội dung website
def scrape_website(website):
    print("Đang chạy trình duyệt...")

    # Sử dụng webdriver_manager để tự động tải và cài đặt ChromeDriver
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")  # Chạy Chrome ở chế độ headless (không mở cửa sổ trình duyệt)
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    try:
        driver.get(website)
        print("Đang tải trang...")
        time.sleep(5)  # Chờ để trang tải xong

        # Lấy nội dung HTML của trang web
        html = driver.page_source
        return html
    finally:
        driver.quit()

# Hàm trích xuất nội dung body của trang web
def extract_body_content(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    body_content = soup.body
    if body_content:
        return str(body_content)
    return ""

# Hàm làm sạch nội dung body
def clean_body_content(body_content):
    soup = BeautifulSoup(body_content, "html.parser")

    # Loại bỏ các thẻ script và style
    for script_or_style in soup(["script", "style"]):
        script_or_style.extract()

    # Lấy văn bản thuần từ nội dung body
    cleaned_content = soup.get_text(separator="\n")
    cleaned_content = "\n".join(
        line.strip() for line in cleaned_content.splitlines() if line.strip()
    )

    return cleaned_content

# Hàm chia nội dung DOM thành các phần nhỏ
def split_dom_content(dom_content, max_length=6000):
    return [
        dom_content[i: i + max_length] for i in range(0, len(dom_content), max_length)
    ]

# Hàm phân tích nội dung với mô hình Ollama
def parse_with_ollama(dom_chunks, parse_description):
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model

    parsed_results = []

    for i, chunk in enumerate(dom_chunks, start=1):
        response = chain.invoke(
            {"dom_content": chunk, "parse_description": parse_description}
        )
        print(f"Parsed batch: {i} of {len(dom_chunks)}")
        parsed_results.append(response)

    return "\n".join(parsed_results)