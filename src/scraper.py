"""
Simple domain-limited crawler for kychub.com.
This saves data/raw_pages.jsonl where each line is:
  {"url": "...", "title": "...", "text": "...", "status": 200}
"""
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from collections import deque
import json, os, time
from tqdm import tqdm

ROOT = "https://www.kychub.com"
OUT = os.path.join("data", "raw_pages.jsonl")
MAX_PAGES = 1000
SLEEP = 0.3

def same_domain(url):
    try:
        return urlparse(url).netloc.endswith("kychub.com")
    except:
        return False

def normalize(url):
    p = urlparse(url)
    normalized = p._replace(fragment="").geturl()
    if normalized.endswith("/"):
        normalized = normalized[:-1]
    return normalized

def extract_text(html):
    soup = BeautifulSoup(html, "html.parser")
    for s in soup(["script", "style", "noscript"]):
        s.decompose()
    main = soup.find("main")
    texts = []
    if main:
        texts.append(main.get_text(separator="\n", strip=True))
    texts.append(soup.get_text(separator="\n", strip=True))
    return "\n\n".join([t for t in texts if t])

def collect_links(html, base):
    soup = BeautifulSoup(html, "html.parser")
    links = set()
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith("mailto:") or href.startswith("tel:"):
            continue
        full = urljoin(base, href)
        if same_domain(full):
            links.add(normalize(full))
    return links

def crawl(start=ROOT):
    visited = set()
    q = deque([normalize(start)])
    results = []
    pbar = tqdm(total=MAX_PAGES, desc="crawled")
    while q and len(visited) < MAX_PAGES:
        url = q.popleft()
        if url in visited:
            continue
        try:
            resp = requests.get(url, timeout=15, headers={"User-Agent":"RAG-Scraper/1.0"})
            if resp.status_code != 200:
                visited.add(url); pbar.update(1); continue
            html = resp.text
            text = extract_text(html)
            if len(text.strip()) < 50:
                visited.add(url); pbar.update(1); continue
            item = {"url": url, "title": BeautifulSoup(html, "html.parser").title.string if BeautifulSoup(html, "html.parser").title else "", "text": text, "status": resp.status_code}
            results.append(item)
            visited.add(url)
            pbar.update(1)
            for l in collect_links(html, url):
                if l not in visited:
                    q.append(l)
            time.sleep(SLEEP)
        except Exception:
            visited.add(url); pbar.update(1); continue
    pbar.close()
    return results

def save_jsonl(items, out=OUT):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"Saved {len(items)} pages to {out}")

if __name__ == "__main__":
    items = crawl()
    save_jsonl(items)