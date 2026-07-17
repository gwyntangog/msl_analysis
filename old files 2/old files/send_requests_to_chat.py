from selenium import webdriver
from selenium.webdriver.common.by import By
import time

driver = webdriver.Chrome()
driver.get("https://chat.openai.com")

time.sleep(20)  # manual login

# send prompt
input_box = driver.find_element(By.TAG_NAME, "textarea")
input_box.send_keys("Hello, summarize copper markets")
input_box.send_keys("\n")

# wait for response to generate
time.sleep(15)

# grab all message bubbles
messages = driver.find_elements(By.CSS_SELECTOR, "div.markdown")

# last one is usually the assistant response
response_text = messages[-1].text

print("RESPONSE:\n", response_text)

# save to file
with open("chatgpt_response.txt", "w", encoding="utf-8") as f:
    f.write(response_text)
