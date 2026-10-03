"""HTTPS with IPv4-first connection attempts and bounded per-address waits."""

import http.client
import socket
import urllib.request


def connect(address, timeout=15, source_address=None, *args, **kwargs):
    host, port = address
    addresses = socket.getaddrinfo(host, port, 0, socket.SOCK_STREAM)
    addresses.sort(key=lambda entry: entry[0] != socket.AF_INET)
    error = None
    for family, kind, protocol, _, target in addresses:
        stream = socket.socket(family, kind, protocol)
        try:
            stream.settimeout(min(timeout, 10))
            if source_address:
                stream.bind(source_address)
            stream.connect(target)
            stream.settimeout(timeout)
            return stream
        except OSError as caught:
            error = caught
            stream.close()
    raise error or OSError("No address found for download host")


class HTTPSConnection(http.client.HTTPSConnection):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._create_connection = connect


class HTTPSHandler(urllib.request.HTTPSHandler):
    def https_open(self, request):
        return self.do_open(HTTPSConnection, request, context=self._context)


def open_url(request, timeout=15):
    return urllib.request.build_opener(HTTPSHandler()).open(request, timeout=timeout)
