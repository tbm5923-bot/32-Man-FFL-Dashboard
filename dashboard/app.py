from pathlib import Path
import base64, lzma

_payload = Path(__file__).with_name("app_source.b85").read_text(encoding="ascii")
_source = lzma.decompress(base64.b85decode(_payload.encode("ascii"))).decode("utf-8")
exec(compile(_source, str(Path(__file__).with_name("app_source.py")), "exec"))
