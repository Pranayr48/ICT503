from __future__ import annotations

import ipaddress
import threading
from collections import deque
from datetime import datetime, timezone
from typing import Any


class NetworkCapture:
    """Small process-local packet capture service for the Streamlit dashboard."""

    def __init__(self) -> None:
        self._packets: deque[dict[str, Any]] = deque(maxlen=500)
        self._lock = threading.Lock()
        self._sniffer = None
        self.error: str | None = None

    @property
    def available(self) -> bool:
        try:
            import scapy.all  # noqa: F401
        except ImportError:
            return False
        return True

    @property
    def running(self) -> bool:
        return self._sniffer is not None and getattr(self._sniffer, "running", False)

    def start(self, interface: str | None = None) -> None:
        try:
            from scapy.all import AsyncSniffer

            self.error = None
            self._sniffer = AsyncSniffer(
                iface=interface or None,
                filter="ip",
                prn=self._on_packet,
                store=False,
            )
            self._sniffer.start()
        except Exception as exc:  # Scapy surfaces OS/Npcap errors here.
            self._sniffer = None
            self.error = str(exc)
            raise RuntimeError(self.error) from exc

    def stop(self) -> None:
        if self._sniffer is not None:
            self._sniffer.stop()
            self._sniffer = None

    def clear(self) -> None:
        with self._lock:
            self._packets.clear()

    def recent(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._packets)[-limit:]

    def _on_packet(self, packet: Any) -> None:
        try:
            from scapy.layers.inet import IP, TCP, UDP

            if not packet.haslayer(IP):
                return
            ip_layer = packet[IP]
            protocol = "TCP" if packet.haslayer(TCP) else "UDP" if packet.haslayer(UDP) else str(ip_layer.proto)
            transport = packet[TCP] if packet.haslayer(TCP) else packet[UDP] if packet.haslayer(UDP) else None
            source_ip = str(ip_layer.src)
            destination_ip = str(ip_layer.dst)
            event = {
                "timestamp": datetime.now(timezone.utc).replace(tzinfo=None),
                "src_ip": source_ip,
                "dst_ip": destination_ip,
                "src_port": int(getattr(transport, "sport", 0)),
                "dst_port": int(getattr(transport, "dport", 0)),
                "protocol": protocol,
                "bytes_sent": int(len(packet)),
                "bytes_received": 0,
                "user_agent": "",
                "url": "",
                "is_internal_traffic": _is_private(source_ip) and _is_private(destination_ip),
                "label": pd_na(),
                "attack_type": "unlabeled",
            }
            with self._lock:
                self._packets.append(event)
        except Exception:
            # A malformed packet should not terminate the capture thread.
            return


def _is_private(value: str) -> bool:
    try:
        return ipaddress.ip_address(value).is_private
    except ValueError:
        return False


def pd_na():
    # Keep the capture module dependency-free; pandas converts this to a missing value.
    return None


_capture = NetworkCapture()


def get_capture() -> NetworkCapture:
    return _capture
