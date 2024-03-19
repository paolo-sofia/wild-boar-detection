import requests
from bs4 import BeautifulSoup

URL = r"https://www.google.com/search?q={}&sca_esv=cb762172827505d3&hl=it&tbm=isch&source=hp&biw=1064&bih=832&ei=DVH5ZcaBC9KVxc8P6vaL2Ac&iflsig=ANes7DEAAAAAZflfHUH6rukl-ebk7MJH4TthUIUUjMO7&ved=0ahUKEwiGvP2n-P-EAxXSSvEDHWr7AnsQ4dUDCAc&uact=5&oq=porcodio&gs_lp=EgNpbWciCHBvcmNvZGlvMgYQABgFGB4yBxAAGIAEGBgyBxAAGIAEGBgyBxAAGIAEGBhI0Q5QowZYrQxwAHgAkAEAmAFPoAG8BKoBATi4AQPIAQD4AQGKAgtnd3Mtd2l6LWltZ5gCCKAC1ASoAgDCAgUQABiABMICCBAAGIAEGLEDwgIEEAAYHsICCRAAGIAEGBgYCpgDA5IHATigB6ge&sclient=img"


def make_google_images_search(search_term: str) -> BeautifulSoup:
    response: requests.Response = requests.get(URL.format(search_term))
    print(response.text)
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def main() -> None:
    search_term = "flower"
    soup: BeautifulSoup = make_google_images_search(search_term)
    print(soup.prettify())


if __name__ == "__main__":
    main()
