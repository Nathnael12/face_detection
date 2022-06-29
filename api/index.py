import socketserver
from http.server import BaseHTTPRequestHandler

from urllib3 import HTTPResponse

def some_function():
    print ("some_function got called")

class MyHandler(BaseHTTPRequestHandler):
    socketserver.TCPServer.allow_reuse_address=True
    def do_GET(self):
        if self.path == '/captureImage':
            # Insert your code here
            some_function()
        self.send_response(200)
        return HTTPResponse("True",preload_content=True)
httpd = socketserver.TCPServer(("", 8080), MyHandler)
httpd.serve_forever()