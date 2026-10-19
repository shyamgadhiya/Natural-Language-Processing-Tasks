import re
text = "Contact us at info@example.com or sales@example.org"
pattern = r"[\w.+-]+@[\w-]+\.[\w.-]+"
emails = re.findall(pattern, text)
print("Emails found:", emails)
