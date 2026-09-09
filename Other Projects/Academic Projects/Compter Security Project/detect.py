import re

def is_ip(url):
    return re.match(r"http[s]?://\d+.\d+.\d+.\d+", url)

def has_suspicious_words(url):
    keywords = ["login", "verify", "secure", "account", "bank"]
    return any(word in url.lower() for word in keywords)

def url_length(url):
    return len(url)

def count_subdomains(url):
    return url.count('.')

def has_special_chars(url):
    return '@' in url or '-' in url

def detect_phishing(url):
    score = 0

    if is_ip(url):
        score += 2
    if has_suspicious_words(url):
        score += 2
    if url_length(url) > 75:
        score += 1
    if count_subdomains(url) > 3:
        score += 1
    if has_special_chars(url):
        score += 1

    if score >= 4:
        return "Likely Phishing"
    elif score >= 2:
        return "Suspicious"
    else:
        return "Safe"

url = input("Enter URL: ")
result = detect_phishing(url)
print("Result:", result)