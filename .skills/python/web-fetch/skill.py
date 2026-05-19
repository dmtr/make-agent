from make_agent import target
import urllib.request
import urllib.error

@target
def fetch_page(url: str) -> str:
    """Fetch the HTML content of the given URL.

    :param url: The web address to retrieve.
    :return: The page content as a string, or an error message if the request fails.
    """
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            charset = response.headers.get_content_charset() or 'utf-8'
            return response.read().decode(charset, errors='replace')
    except urllib.error.HTTPError as e:
        return f"HTTP error {e.code} fetching {url}: {e.reason}"
    except urllib.error.URLError as e:
        return f"URL error fetching {url}: {e.reason}"
    except Exception as e:
        return f"Error fetching {url}: {e}"
