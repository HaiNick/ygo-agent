#!/usr/bin/env python3
"""
YGOPro packet-aware WebSocket <-> TCP bridge.
Bridges ws://127.0.0.1:7912 (Neos browser client) to 127.0.0.1:7911 (ygopro-server TCP),
ensuring every YGOPro packet ([uint16_le length][payload]) is delivered as a complete
binary WebSocket frame without mid-packet TCP fragmentation.
"""
import argparse
import asyncio
import logging
import struct
import websockets

logging.basicConfig(level=logging.INFO, format="%(asctime)s [ws_tcp_proxy] %(message)s")


async def handle_client(ws, tcp_host: str, tcp_port: int):
    peer = ws.remote_address
    logging.info(f"WebSocket client connected from {peer} -> bridging to {tcp_host}:{tcp_port}")
    try:
        reader, writer = await asyncio.open_connection(tcp_host, tcp_port)
    except Exception as e:
        logging.error(f"Failed to connect to TCP {tcp_host}:{tcp_port}: {e}")
        await ws.close()
        return

    async def ws_to_tcp():
        try:
            async for message in ws:
                if isinstance(message, bytes):
                    writer.write(message)
                    await writer.drain()
        except Exception as e:
            logging.debug(f"ws_to_tcp closed: {e}")
        finally:
            try:
                writer.close()
            except Exception:
                pass

    async def tcp_to_ws():
        try:
            while True:
                hdr = await reader.readexactly(2)
                (pkt_len,) = struct.unpack("<H", hdr)
                body = await reader.readexactly(pkt_len)
                await ws.send(hdr + body)
        except asyncio.IncompleteReadError:
            logging.info(f"TCP server closed connection for {peer}")
        except Exception as e:
            logging.debug(f"tcp_to_ws closed: {e}")
        finally:
            try:
                await ws.close()
            except Exception:
                pass

    await asyncio.gather(ws_to_tcp(), tcp_to_ws(), return_exceptions=True)
    logging.info(f"Session ended for {peer}")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ws-host", default="127.0.0.1")
    parser.add_argument("--ws-port", type=int, default=7912)
    parser.add_argument("--tcp-host", default="127.0.0.1")
    parser.add_argument("--tcp-port", type=int, default=7911)
    args = parser.parse_args()

    logging.info(f"Listening on ws://{args.ws_host}:{args.ws_port} -> tcp://{args.tcp_host}:{args.tcp_port}")
    async with websockets.serve(
        lambda ws: handle_client(ws, args.tcp_host, args.tcp_port),
        args.ws_host,
        args.ws_port,
        max_size=None,
        ping_interval=None,
    ):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
