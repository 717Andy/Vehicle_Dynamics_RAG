import requests
from bs4 import BeautifulSoup

def fetch_and_extract(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            content = soup.find('div', {'id': 'mw-content-text'})
            
            # Find all relevant text containers
            # We look for paragraphs OR the lists that often follow them
            elements = content.find_all(['p', 'ul'])
            
            extracted_text = []
            for el in elements:
                # Only include lists if they are direct siblings of a <p> 
                # or within the main content flow
                text = el.get_text(separator=' ', strip=True)
                if text:
                    extracted_text.append(text)
            
            # Join with double newlines to keep chunks distinct
            full_text = "\n\n".join(extracted_text)
            
            with open('Selected_Document.txt', 'w', encoding='utf-8') as f:
                f.write(full_text)
            
            print(f"Success: Content extracted from {url}")
            return full_text
        else:
            print(f"Failure: Received status code {response.status_code}")
            return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    

def main():
    url = "https://en.wikipedia.org/wiki/Vehicle_dynamics"
    fetch_and_extract(url)

if __name__ == '__main__':
    main()