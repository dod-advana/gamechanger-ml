from attr import has
import requests
import os
import base64
import hmac, hashlib


class GCWebClient:
    def __init__(
        self,
        host: str = os.environ.get("GC_WEB_HOST", "localhost"),
        port: str = os.environ.get("GC_WEB_PORT", 8990),
        token: str = os.environ.get("ML_WEB_TOKEN",""),
        gc_enable_ssl: str = os.environ.get(
            "GC_ENABLE_SSL", "True").lower() == "true",
        ):
        self.host = host
        self.port = port
        self.enable_ssl = gc_enable_ssl
        self.token = token

    @property
    def getURL(self):
        if self.enable_ssl:
            url = "https://"
        else:
            url = "http://"
        url = f"{url}{self.host}:{self.port}"
        return url

    def getHeader(self,hash):
        return {
            "X-UA-SIGNATURE":hash,
            "SSL_CLIENT_S_DN_CN":"ml-api",
        }
    def getHash(self,endpoint):
        h = hmac.new(self.token.encode(), digestmod=hashlib.sha256)
        h.update(endpoint.encode())
        hash = base64.b64encode(h.digest())
        return hash

    def getSearchMappings(self, daysBack=3):
        #endpoint needs to be hashed before adding query params since thats how the web backend calculates the hash
        endpoint = f"/api/gameChanger/admin/getSearchPdfMapping"
        hash = self.getHash(endpoint)
        endpoint += f'?daysBack={daysBack}'
        r = requests.get(self.getURL + endpoint, headers=self.getHeader(hash))
        return r.content
