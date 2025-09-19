import os
import webbrowser
import http.server
import socketserver

# Navigate to the Model1_5 directory
os.chdir(r"D:\Work\SIH_ML\Model1_5")

class CORSHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

PORT = 8080

print("🌐 Starting Ocean Hazard Test Frontend...")
print(f"📁 Serving from: {os.getcwd()}")
print(f"🔗 URL: http://localhost:{PORT}/test_frontend.html")
print("\n🚀 Make sure to start your Model 1.5 API first:")
print("   1. Start Docker Desktop")
print("   2. Run: docker-compose up -d")
print("   3. API will be at: http://localhost:8000")
print("\n💡 Press Ctrl+C to stop\n")

with socketserver.TCPServer(("", PORT), CORSHTTPRequestHandler) as httpd:
    # Open browser
    webbrowser.open(f'http://localhost:{PORT}/test_frontend.html')
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 Test server stopped!")