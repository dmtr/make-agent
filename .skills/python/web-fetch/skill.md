---
description: "Fetch a web page"
---

# Web Fetch Skill

This skill provides a tool to fetch the contents of a web page given its URL.

## Usage

Call the `fetch_page` function with a URL string to retrieve the page's HTML content.

```python
result = execute_skill(name="web-fetch", target="fetch_page", kwargs={"url": "https://example.com"})
```

The function returns the page content as a string, or an error message if the request fails.

## Available tools

- `fetch_page(url: str)` — Retrieves the HTML content of the specified URL.

---

**Note**: The function follows redirects and uses a reasonable timeout.
