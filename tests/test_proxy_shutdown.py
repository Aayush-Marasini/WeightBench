"""Proxy teardown must finish decision logging and cancel idle client/tunnel sockets."""
import json
import socket
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from weightbench.harness import audit, sandbox


def client_for(proxy):
    client = socket.socket()
    client.settimeout(5)
    client.connect(proxy.server_address)
    return client


@pytest.mark.parametrize("host", ["denied.example", "api.anthropic.com"])
def test_stop_waits_for_decision_before_egress_audit(tmp_path, monkeypatch, host):
    entered, release, stopping, stopped = (threading.Event() for _ in range(4))
    proxy = sandbox.EgressProxy(tmp_path / "egress.jsonl").start()
    log, shutdown = proxy.log, proxy.shutdown

    def held_log(*args):
        entered.set()
        assert release.wait(10)
        log(*args)

    def stopping_accepts():
        shutdown()
        stopping.set()

    def stop():
        proxy.stop()
        stopped.set()

    monkeypatch.setattr(proxy, "log", held_log)
    monkeypatch.setattr(proxy, "shutdown", stopping_accepts)
    dials = []

    def dial(*args, **kwargs):
        dials.append(args)
        raise OSError("dial should not start after the stopping flag is observed")

    monkeypatch.setattr(sandbox.socket, "create_connection", dial)
    client = client_for(proxy)
    client.sendall(f"CONNECT {host}:443 HTTP/1.1\r\n\r\n".encode())
    with ThreadPoolExecutor(1) as pool:
        cleanup = None
        try:
            assert entered.wait(10)
            cleanup = pool.submit(stop)
            assert stopping.wait(10)
            assert not stopped.wait(.1), "proxy stopped while a decision could still change its audit"
        finally:
            release.set()
            client.close()
            if cleanup is None:
                proxy.stop()
        cleanup.result(timeout=10)
    summary = audit.egress_summary(proxy.log_path)
    if host == "denied.example":
        assert summary == {"allowed": 0, "denied": ["denied.example:443"], "n_denied": 1}
    else:
        assert summary == {"allowed": 1, "denied": [], "n_denied": 0}
    assert not dials


@pytest.mark.parametrize("headers", [b"", b"CONNECT denied.example:443 HTTP/1.1\r\nHost:"])
def test_stop_cancels_idle_and_partial_header_clients(tmp_path, monkeypatch, headers):
    entered = threading.Event()
    proxy = sandbox.EgressProxy(tmp_path / "egress.jsonl")
    finish = proxy.finish_request

    def admitted(*args):
        entered.set()
        finish(*args)

    monkeypatch.setattr(proxy, "finish_request", admitted)
    proxy.start()
    client = client_for(proxy)
    if headers:
        client.sendall(headers)
    with ThreadPoolExecutor(1) as pool:
        cleanup = None
        try:
            assert entered.wait(10)
            cleanup = pool.submit(proxy.stop)
            cleanup.result(timeout=5)
            assert client.recv(4096) == b""
        finally:
            client.close()
            if cleanup is None:
                proxy.stop()
    if headers:
        assert audit.egress_summary(proxy.log_path)["denied"] == ["denied.example:443"]
    else:
        assert audit.egress_summary(proxy.log_path) == {"allowed": 0, "denied": [], "n_denied": 0}


def test_stop_closes_established_allowed_tunnel(tmp_path, monkeypatch):
    upstream_closed = threading.Event()
    echo = socket.socket()
    echo.bind(("127.0.0.1", 0))
    echo.listen(1)
    echo.settimeout(5)

    def serve():
        connection, _ = echo.accept()
        with connection:
            connection.settimeout(5)
            assert connection.recv(4) == b"ping"
            connection.sendall(b"pong")
            assert connection.recv(1) == b""
        upstream_closed.set()

    serving = threading.Thread(target=serve, daemon=True)
    serving.start()
    connect = socket.create_connection
    monkeypatch.setattr(sandbox.socket, "create_connection",
                        lambda address, timeout=None: connect(echo.getsockname(), timeout=timeout))
    proxy = sandbox.EgressProxy(tmp_path / "egress.jsonl").start()
    client = client_for(proxy)
    try:
        client.sendall(b"CONNECT api.anthropic.com:443 HTTP/1.1\r\n\r\n")
        assert b" 200 " in client.recv(4096)
        client.sendall(b"ping")
        assert client.recv(4) == b"pong"
        before = proxy.log_path.read_bytes()
        with ThreadPoolExecutor(1) as pool:
            cleanup = pool.submit(proxy.stop)
            try:
                cleanup.result(timeout=5)
            finally:
                client.close()
        assert upstream_closed.wait(5)
        assert proxy.log_path.read_bytes() == before
        assert json.loads(before)["allowed"] is True
    finally:
        client.close()
        proxy.stop()
        echo.close()
        serving.join(timeout=5)
