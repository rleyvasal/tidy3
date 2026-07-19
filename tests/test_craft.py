"""Remote-seeding payload and bootstrap plumbing (no kernel needed)."""

from __future__ import annotations

import base64
import io
import tarfile

from tidy3 import __version__, craft


def test_payload_roundtrip_contains_package():
    payload, stamp = craft.build_payload()
    raw = base64.b64decode(payload)
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
        names = tar.getnames()
    assert "tidy3/__init__.py" in names
    assert "tidy3/frame.py" in names
    assert "tidy3/verbs.py" in names
    assert "tidy3/tidyselect.py" in names
    assert "tidy3/join_spec.py" in names
    assert "tidy3/bench_suite.py" in names
    assert "tidy3/craft.py" in names
    assert stamp.startswith(__version__ + "-")


def test_stamp_is_stable_across_builds():
    _, s1 = craft.build_payload()
    _, s2 = craft.build_payload()
    assert s1 == s2


def test_bootstrap_is_valid_python_with_stamp_and_sentinel():
    payload, stamp = craft.build_payload()
    code = craft.bootstrap_code(payload, stamp)
    assert stamp in code
    assert "tidy3 remote: OK" in code
    compile(code, "<tidy3.bootstrap>", "exec")
    # style flag off must also produce valid code
    compile(
        craft.bootstrap_code(payload, stamp, style_polars=False),
        "<tidy3.bootstrap>",
        "exec",
    )


def test_seed_parses_success_failure_and_exception():
    ok, msg = craft.seed(
        lambda code, max_chars=0: "installing...\ntidy3 remote: OK v9.9 (x)\n"
    )
    assert ok
    assert "tidy3 remote: OK" in msg

    ok, msg = craft.seed(lambda code, max_chars=0: "Traceback ...\nboom\n")
    assert not ok
    assert "boom" in msg

    def raiser(code, max_chars=0):
        raise RuntimeError("kernel down")

    ok, msg = craft.seed(raiser)
    assert not ok
    assert "kernel down" in msg
