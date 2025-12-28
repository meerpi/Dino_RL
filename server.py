"""
HTTP Server for T-Rex Game
Serves the game with proper CORS headers to avoid "tainted canvas" errors
"""

import http.server
import socketserver
import os
import sys

PORT = 8000

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Custom HTTP handler with CORS headers"""
    
    def end_headers(self):
        # Add CORS headers to allow canvas access
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        super().end_headers()
    
    def log_message(self, format, *args):
        """Custom logging to reduce noise"""
        # Only log non-resource requests
        # Convert args to string to handle HTTPStatus enums
        try:
            if args:
                first_arg = str(args[0]) if args[0] else ""
                if not any(x in first_arg for x in ['.png', '.jpg', '.css', 'favicon', '.well-known']):
                    super().log_message(format, *args)
        except:
            # If anything goes wrong, just use default logging
            super().log_message(format, *args)


def main():
    """Start the HTTP server"""
    # Change to the t-rex-runner directory where index.html is located
    game_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 't-rex-runner')
    
    if not os.path.exists(game_dir):
        print(f"❌ ERROR: Game directory not found: {game_dir}")
        print(f"   Make sure 't-rex-runner' folder exists in: {os.path.dirname(os.path.abspath(__file__))}")
        sys.exit(1)
    
    os.chdir(game_dir)
    
    # Check for index.html
    if not os.path.exists('index.html'):
        print(f"❌ ERROR: index.html not found in {game_dir}")
        print(f"   Files in directory: {os.listdir('.')}")
        sys.exit(1)
    
    print("=" * 70)
    print("T-REX GAME HTTP SERVER")
    print("=" * 70)
    print(f"📁 Serving files from: {os.getcwd()}")
    print(f"📄 Files in directory:")
    for f in sorted(os.listdir('.'))[:15]:  # Show first 15 files
        print(f"   - {f}")
    
    if len(os.listdir('.')) > 15:
        print(f"   ... and {len(os.listdir('.')) - 15} more files")
    print()
    
    try:
        with socketserver.TCPServer(("", PORT), MyHTTPRequestHandler) as httpd:
            print("=" * 70)
            print(f"🌐 Server running at http://localhost:{PORT}")
            print(f"👉 Open http://localhost:{PORT}/index.html in your browser")
            print()
            print("⚠️  IMPORTANT STARTUP SEQUENCE:")
            print("   1. ✓ HTTP server is running (this window)")
            print("   2. Run test script: python test_env.py")
            print("   3. THEN open browser to http://localhost:8000/index.html")
            print("   4. The test script creates the WebSocket server!")
            print()
            print("   Note: The browser will show WebSocket errors until step 2 is done.")
            print("=" * 70)
            print("\nServer logs:")
            print("-" * 70)
            
            httpd.serve_forever()
            
    except KeyboardInterrupt:
        print("\n\n" + "=" * 70)
        print("Server stopped by user")
        print("=" * 70)
    except OSError as e:
        if e.errno == 48 or e.errno == 98:  # Address already in use
            print(f"\n❌ ERROR: Port {PORT} is already in use!")
            print("   Either:")
            print("   1. Stop the other process using this port")
            print("   2. Change PORT variable in this script")
        else:
            print(f"\n❌ ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()