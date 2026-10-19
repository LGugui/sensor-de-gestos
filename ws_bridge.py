"""
BigBang WebSocket Bridge
Envia landmarks de mãos em tempo real para o browser (ws://localhost:8765).
Uso: python ws_bridge.py
"""

import asyncio
import json
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(__file__))

try:
    import websockets
except ImportError:
    print("[BigBang Bridge] Instalando websockets...")
    os.system(f"{sys.executable} -m pip install websockets")
    import websockets

from detector import HandDetector

PORT = 8765

CFG = {
    "max_hands": 2,
    "detection_confidence": 0.6,
    "tracking_confidence": 0.5,
    "flip_handedness": True,
    "visibility_threshold": 0.4,
}

clients: set = set()


async def handler(ws):
    clients.add(ws)
    print(f"[Bridge] Cliente conectado — {len(clients)} ativo(s)")
    try:
        await ws.wait_closed()
    finally:
        clients.discard(ws)
        print(f"[Bridge] Cliente desconectado — {len(clients)} ativo(s)")


async def camera_loop():
    detector = HandDetector(CFG)

    # DirectShow evita conflito com getUserMedia do browser no Windows
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        # Fallback sem backend específico
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[Bridge] ERRO: câmera não disponível.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print(f"[Bridge] Câmera ativa — enviando para ws://localhost:{PORT}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                await asyncio.sleep(0.01)
                continue

            frame = cv2.flip(frame, 1)
            _, hands = detector.find_hands(frame, draw=False)

            if clients:
                payload = []
                for lm, label in hands:
                    payload.append({
                        "label": label,
                        "landmarks": [
                            {"x": round(l.x, 4), "y": round(l.y, 4), "z": round(l.z, 4)}
                            for l in lm
                        ],
                    })
                msg = json.dumps({"hands": payload})
                dead = set()
                for c in clients:
                    try:
                        await c.send(msg)
                    except Exception:
                        dead.add(c)
                clients -= dead

            await asyncio.sleep(1 / 60)
    finally:
        cap.release()
        detector.close()
        print("[Bridge] Câmera liberada.")


async def main():
    print(f"[BigBang Bridge] Iniciando ws://localhost:{PORT}")
    print("[BigBang Bridge] Ctrl+C para parar\n")
    async with websockets.serve(handler, "localhost", PORT):
        await camera_loop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[Bridge] Encerrado.")
