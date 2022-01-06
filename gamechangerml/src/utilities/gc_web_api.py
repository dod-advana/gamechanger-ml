import requests
import os


class GCWebClient:
    def __init__(
        self,
        host: str = os.environ.get("GC_WEB_HOST", "localhost"),
        port: str = os.environ.get("GC_WEB_PORT", 8990),
        enable_ssl: str = os.environ.get(
            "ES_ENABLE_SSL", "True").lower() == "true",
    ):
        self.host = host
        self.port = port
        self.user = "steve"
        self.enable_ssl = enable_ssl

    @property
    def getURL(self):
        if self.enable_ssl:
            url = "https://"
        else:
            url = "http://"
        url = f"{url}{self.host}:{self.port}/"
        return url

    @property
    def getHeader(self):
        return {"ssl_client_s_dn_cn": self.user}

    def getSearchMappings(self, daysBack=3):
        endpoint = f"api/gameChanger/admin/getSearchPdfMapping?daysBack={daysBack}"
        r = requests.get(self.getURL + endpoint, headers=self.getHeader)
        return r.content
