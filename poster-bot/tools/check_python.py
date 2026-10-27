"""Подходит ли этот Python: нейросети (onnxruntime) есть только для 64-битного Python 3.11–3.14.
Код выхода 0 — подходит, 1 — нет. Печатает версию."""
import struct
import sys

ok = (3, 11) <= sys.version_info[:2] <= (3, 14) and struct.calcsize("P") == 8 \
    and getattr(sys, "_is_gil_enabled", lambda: True)()
print(f"Python {sys.version.split()[0]} ({struct.calcsize('P') * 8}-bit)")
sys.exit(0 if ok else 1)
