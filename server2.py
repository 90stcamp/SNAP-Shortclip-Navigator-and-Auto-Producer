from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import subprocess

class RequestHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        print(self.path)
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        data = json.loads(post_data.decode('utf-8'))

        print("Received data:", data)
        
        cmd1 = f"sed -i 's|YOUTUBE_LINK=.*|YOUTUBE_LINK=https://www.youtube.com/watch?v={data['youtube_link']}|' .env"
        cmd2 = f"sed -i 's|YOUTUBE_CATEGORY=.*|YOUTUBE_CATEGORY={data['category']}|' .env"


        subprocess.call(cmd1, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.call(cmd2, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        response_message = {"message": "Data received successfully"}
        self.wfile.write(json.dumps(response_message).encode('utf-8'))

def run(server_class=HTTPServer, handler_class=RequestHandler, port=8088):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f"Server B is running on port {port}")
    httpd.handle_request()

if __name__ == '__main__':
    run(port=8088)