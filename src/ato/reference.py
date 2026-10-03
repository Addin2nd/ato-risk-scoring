"""Static reference data: cities, autonomous systems (ASNs) and browser/device profiles.

In production this layer is replaced by real enrichment sources (MaxMind GeoIP,
IP-to-ASN feeds, a hosting/VPN IP reputation list and a user-agent parser).
"""
from __future__ import annotations

import numpy as np

# name, region, country ISO-2, lat, lon
CITIES: dict[str, tuple[str, str, float, float]] = {
    # Indonesia
    "Jakarta": ("DKI Jakarta", "ID", -6.2088, 106.8456),
    "Bandung": ("Jawa Barat", "ID", -6.9175, 107.6191),
    "Surabaya": ("Jawa Timur", "ID", -7.2575, 112.7521),
    "Medan": ("Sumatera Utara", "ID", 3.5952, 98.6722),
    "Semarang": ("Jawa Tengah", "ID", -6.9667, 110.4167),
    "Yogyakarta": ("DI Yogyakarta", "ID", -7.7956, 110.3695),
    "Makassar": ("Sulawesi Selatan", "ID", -5.1477, 119.4327),
    "Denpasar": ("Bali", "ID", -8.6705, 115.2126),
    "Palembang": ("Sumatera Selatan", "ID", -2.9761, 104.7754),
    "Balikpapan": ("Kalimantan Timur", "ID", -1.2379, 116.8529),
    "Malang": ("Jawa Timur", "ID", -7.9666, 112.6326),
    "Tangerang": ("Banten", "ID", -6.1783, 106.6319),
    "Bekasi": ("Jawa Barat", "ID", -6.2383, 106.9756),
    "Pekanbaru": ("Riau", "ID", 0.5071, 101.4478),
    # Rest of world
    "Singapore": ("Singapore", "SG", 1.3521, 103.8198),
    "Kuala Lumpur": ("Kuala Lumpur", "MY", 3.1390, 101.6869),
    "Bangkok": ("Bangkok", "TH", 13.7563, 100.5018),
    "Tokyo": ("Tokyo", "JP", 35.6762, 139.6503),
    "Sydney": ("New South Wales", "AU", -33.8688, 151.2093),
    "Seoul": ("Seoul", "KR", 37.5665, 126.9780),
    "Jeddah": ("Makkah", "SA", 21.4858, 39.1925),
    "Hong Kong": ("Hong Kong", "HK", 22.3193, 114.1694),
    "Amsterdam": ("North Holland", "NL", 52.3676, 4.9041),
    "Frankfurt": ("Hesse", "DE", 50.1109, 8.6821),
    "London": ("England", "GB", 51.5074, -0.1278),
    "Ashburn": ("Virginia", "US", 39.0438, -77.4874),
    "Los Angeles": ("California", "US", 34.0522, -118.2437),
    "Lagos": ("Lagos", "NG", 6.5244, 3.3792),
    "Moscow": ("Moscow", "RU", 55.7558, 37.6173),
    "Sao Paulo": ("Sao Paulo", "BR", -23.5505, -46.6333),
    "Hanoi": ("Hanoi", "VN", 21.0285, 105.8542),
    "Manila": ("Metro Manila", "PH", 14.5995, 120.9842),
    "Mumbai": ("Maharashtra", "IN", 19.0760, 72.8777),
}

ID_HOME_CITIES = ["Jakarta", "Bandung", "Surabaya", "Medan", "Semarang", "Yogyakarta",
                  "Makassar", "Denpasar", "Palembang", "Balikpapan", "Malang",
                  "Tangerang", "Bekasi", "Pekanbaru"]
ID_HOME_WEIGHTS = np.array([30, 10, 10, 6, 5, 5, 4, 4, 3, 2, 3, 8, 8, 2], dtype=float)
ID_HOME_WEIGHTS /= ID_HOME_WEIGHTS.sum()

# Legit leisure / business / religious travel destinations abroad
TRAVEL_ABROAD = ["Singapore", "Kuala Lumpur", "Bangkok", "Tokyo", "Seoul", "Jeddah",
                 "Sydney", "Hong Kong", "Amsterdam"]

# asn -> (name, type, list of cities where it has presence)
# type: fixed | mobile | hosting
ASNS: dict[int, tuple[str, str, list[str]]] = {
    7713: ("Telkom Indonesia", "fixed", ID_HOME_CITIES),
    17451: ("Biznet Networks", "fixed", ID_HOME_CITIES),
    23700: ("Link Net (First Media)", "fixed", ["Jakarta", "Bandung", "Surabaya", "Tangerang",
                                               "Bekasi", "Medan", "Semarang", "Malang"]),
    23693: ("Telkomsel", "mobile", ID_HOME_CITIES),
    24203: ("XL Axiata", "mobile", ID_HOME_CITIES),
    4761: ("Indosat Ooredoo Hutchison", "mobile", ID_HOME_CITIES),
    7473: ("Singtel", "fixed", ["Singapore"]),
    4788: ("TM Net", "fixed", ["Kuala Lumpur"]),
    7470: ("True Internet", "mobile", ["Bangkok"]),
    2516: ("KDDI", "mobile", ["Tokyo"]),
    4766: ("Korea Telecom", "fixed", ["Seoul"]),
    25019: ("Saudi Telecom (STC)", "mobile", ["Jeddah"]),
    1221: ("Telstra", "mobile", ["Sydney"]),
    4760: ("HKT", "fixed", ["Hong Kong"]),
    1136: ("KPN", "fixed", ["Amsterdam"]),
    3320: ("Deutsche Telekom", "fixed", ["Frankfurt"]),
    2856: ("BT", "fixed", ["London"]),
    7922: ("Comcast", "fixed", ["Ashburn", "Los Angeles"]),
    29465: ("MTN Nigeria", "mobile", ["Lagos"]),
    12389: ("Rostelecom", "fixed", ["Moscow"]),
    28573: ("Claro Brasil", "mobile", ["Sao Paulo"]),
    7552: ("Viettel", "mobile", ["Hanoi"]),
    9299: ("PLDT", "fixed", ["Manila"]),
    55836: ("Reliance Jio", "mobile", ["Mumbai"]),
    # Hosting / VPN egress — the same ranges legit VPN users and attackers share
    14061: ("DigitalOcean", "hosting", ["Singapore", "Amsterdam", "Frankfurt", "London", "Ashburn", "Mumbai"]),
    16276: ("OVH", "hosting", ["Frankfurt", "London", "Amsterdam", "Singapore"]),
    16509: ("Amazon AWS", "hosting", ["Singapore", "Tokyo", "Ashburn", "Frankfurt", "Jakarta"]),
    9009: ("M247", "hosting", ["Singapore", "Amsterdam", "London", "Frankfurt", "Hong Kong", "Los Angeles"]),
    212238: ("Datacamp (CDN77)", "hosting", ["Singapore", "Amsterdam", "Frankfurt", "Ashburn", "Tokyo"]),
    63949: ("Akamai (Linode)", "hosting", ["Singapore", "Tokyo", "Los Angeles", "Mumbai"]),
}

ID_FIXED = [7713, 17451, 23700]
ID_MOBILE = [23693, 24203, 4761]
ID_MOBILE_WEIGHTS = np.array([0.5, 0.28, 0.22])
HOSTING = [a for a, v in ASNS.items() if v[1] == "hosting"]
VPN_PROVIDERS = [9009, 212238, 14061]  # commercial VPN egress commonly sits here


def foreign_consumer_asn(city: str) -> int:
    """Return the consumer (non-hosting) ASN serving a foreign city."""
    for asn, (_, kind, cities) in ASNS.items():
        if kind != "hosting" and city in cities and CITIES[city][1] != "ID":
            return asn
    raise KeyError(city)


# browser, browser_version, os, os_version, device_type
MOBILE_PLATFORMS = [
    ("Chrome Mobile", "128.0", "Android", "14", "mobile"),
    ("Chrome Mobile", "127.0", "Android", "13", "mobile"),
    ("Samsung Internet", "25.0", "Android", "14", "mobile"),
    ("Mobile Safari", "17.5", "iOS", "17.5", "mobile"),
    ("Mobile Safari", "18.0", "iOS", "18.0", "mobile"),
]
MOBILE_WEIGHTS = np.array([0.32, 0.18, 0.15, 0.17, 0.18])
DESKTOP_PLATFORMS = [
    ("Chrome", "128.0", "Windows", "11", "desktop"),
    ("Chrome", "128.0", "Windows", "10", "desktop"),
    ("Edge", "128.0", "Windows", "11", "desktop"),
    ("Firefox", "130.0", "Windows", "10", "desktop"),
    ("Safari", "17.6", "Mac OS X", "14.6", "desktop"),
    ("Chrome", "128.0", "Mac OS X", "14.6", "desktop"),
]
DESKTOP_WEIGHTS = np.array([0.30, 0.18, 0.14, 0.08, 0.15, 0.15])
AUTOMATION_PLATFORMS = [
    ("Python Requests", "2.31", "Other", "", "bot"),
    ("HeadlessChrome", "120.0", "Linux", "", "desktop"),
    ("curl", "8.4", "Other", "", "bot"),
]
AUTOMATION_BROWSERS = {"Python Requests", "HeadlessChrome", "curl"}


def user_agent_string(browser: str, bver: str, os_name: str, os_ver: str, dtype: str) -> str:
    """Render a plausible User-Agent header for a platform tuple."""
    if browser == "Python Requests":
        return f"python-requests/{bver}"
    if browser == "curl":
        return f"curl/{bver}"
    if browser == "HeadlessChrome":
        return (f"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
                f"HeadlessChrome/{bver}.0.0 Safari/537.36")
    if os_name == "Android":
        tag = "SamsungBrowser" if browser == "Samsung Internet" else "Chrome"
        return (f"Mozilla/5.0 (Linux; Android {os_ver}; K) AppleWebKit/537.36 (KHTML, like Gecko) "
                f"{tag}/{bver} Mobile Safari/537.36")
    if os_name == "iOS":
        v = os_ver.replace(".", "_")
        return (f"Mozilla/5.0 (iPhone; CPU iPhone OS {v} like Mac OS X) AppleWebKit/605.1.15 "
                f"(KHTML, like Gecko) Version/{bver} Mobile/15E148 Safari/604.1")
    if os_name == "Mac OS X":
        engine = f"Version/{bver} Safari/605.1.15" if browser == "Safari" else f"Chrome/{bver}.0.0 Safari/537.36"
        return f"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) {engine}"
    nt = "10.0"
    tail = {"Edge": f"Chrome/{bver}.0.0 Safari/537.36 Edg/{bver}.0.0",
            "Firefox": f"Gecko/20100101 Firefox/{bver}"}.get(browser, f"Chrome/{bver}.0.0 Safari/537.36")
    return f"Mozilla/5.0 (Windows NT {nt}; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) {tail}"


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km. Works on scalars or numpy arrays."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))
