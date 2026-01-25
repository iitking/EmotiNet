import uvicorn
import argparse
from app.config import HOST, PORT, DEBUG

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run EmotiNet Emotion Detection Server")
    parser.add_argument("--host", type=str, default=HOST, help="Server host IP")
    parser.add_argument("--port", type=int, default=PORT, help="Server port")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload on code change")

    args = parser.parse_args()

    print(f"✨ Starting EmotiNet on http://{args.host}:{args.port}")
    print(f"📚 Interactive Swagger API Docs: http://{args.host}:{args.port}/docs")
    print(f"🖥️ Web UI Interface: http://{args.host}:{args.port}/")

    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload or DEBUG
    )
