from bs4 import BeautifulSoup
from selenium import webdriver
from selenium_stealth import stealth
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import tempfile
import os
import time
import random
import csv

print("✅ All libraries installed correctly!")

# --- prepare CSV file ---
csv_filename = "yellowpages_results.csv"
with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["Name", "Phone", "Address", "Website 1", "Website 2"])  # header row

# --- pagination loop without proxy rotation ---
base_url = "https://www.yellowpages.com/sunnyvale-ca/plumbers"

for page_num in range(1, 6):  # <-- change 6 to number of pages you want
    print(f"\n=== Scraping page {page_num} ===")

    temp_dir = tempfile.mkdtemp()
    options = Options()
    options.add_argument(f"--user-data-dir={temp_dir}")
    options.add_argument("--start-maximized")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--disable-infobars")
    options.add_argument("--no-sandbox")

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)

    stealth(driver,
            languages=["en-US", "en"],
            vendor="Google Inc.",
            platform="Win32",
            webgl_vendor="Intel Inc.",
            renderer="Intel Iris OpenGL Engine",
            fix_hairline=True,
            )

    if page_num == 1:
        url = base_url
    else:
        url = f"{base_url}?page={page_num}"

    try:
        driver.get(url)
        time.sleep(random.uniform(8, 15))

        page_height = driver.execute_script("return document.body.scrollHeight")
        scroll_pos = 0
        for _ in range(random.randint(3, 6)):
            increment = random.randint(max(1, int(page_height * 0.15)), max(1, int(page_height * 0.4)))
            scroll_pos = min(scroll_pos + increment, page_height)
            driver.execute_script(f"window.scrollTo(0, {scroll_pos});")
            time.sleep(random.uniform(0.6, 2.0))

        time.sleep(random.uniform(2.0, 4.5))

        title = driver.title or ""
        page_source_lower = driver.page_source.lower() if driver.page_source else ""
        if ("attention required" in title.lower() or 
            "sorry, you have been blocked" in page_source_lower or 
            "please enable cookies" in page_source_lower):
            print(f"⚠️ Cloudflare/Block detected on page {page_num} (title: {title}). Saving snapshot and backing off.")
            filename = f"page_{page_num}_blocked.html"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(driver.page_source)
            print(f"Saved blocked-page HTML to: {os.path.abspath(filename)}")
            backoff = random.uniform(60, 180)
            print(f"Waiting {int(backoff)}s before continuing to next page...")
            time.sleep(backoff)
            driver.quit()
            continue

        html_text = driver.page_source
        soup = BeautifulSoup(html_text, 'lxml')
        jobs = soup.find_all('div', class_='result')

        # open CSV file in append mode
        with open(csv_filename, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)

            for job in jobs:
                Names = job.find('a', class_='business-name')
                Name = Names.text.strip() if Names else "N/A"
                Phones = job.find('div', class_=['phones phone primary', 'phone'])
                Phone = Phones.text.strip() if Phones else "N/A"
                Adressess = job.find(['div', 'p'], class_='adr')
                Adress = Adressess.text.strip() if Adressess else "N/A"
                website_link_tag1 = job.find('a', class_='track-visit-website')
                website_link1 = website_link_tag1['href'] if website_link_tag1 else "N/A"
                website_link_tag2 = job.find('div', class_='links')
                if website_link_tag2:
                    first_a_tag = website_link_tag2.find('a')
                    website_link2 = first_a_tag['href'] if first_a_tag and first_a_tag.has_attr('href') else "N/A"
                else:
                    website_link2 = "N/A"

                # print to console
                print(Name)
                print(Phone)
                print(Adress)
                print(website_link1)
                print(website_link2)
                print("--" * 50)

                # save to CSV
                writer.writerow([Name, Phone, Adress, website_link1, website_link2])

    except Exception as e:
        print(f"❌ Error scraping page {page_num}: {e}")

    finally:
        try:
            driver.quit()
        except Exception:
            pass

print(f"✅ Done scraping pages. Data saved to '{csv_filename}'.")
