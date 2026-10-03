"""Preserve complete JPEG XMP packets and referenced auxiliary image data."""

import struct
from pathlib import Path

XMP_HEADERS = (b"http://ns.adobe.com/xap/1.0/\0", b"http://ns.adobe.com/xmp/extension/\0")


def segments(data):
    """Return marker spans and the primary image end, including progressive scans."""
    if not data.startswith(b"\xff\xd8"):
        raise ValueError("Invalid JPEG header")
    result = []
    position = 2
    while position < len(data):
        start = position
        if data[position] != 255:
            raise ValueError("Invalid JPEG marker")
        while position < len(data) and data[position] == 255:
            position += 1
        if position >= len(data):
            break
        marker = data[position]
        position += 1
        if marker == 0xD9:
            return result, position
        if marker in (0xD8, 0x01, *range(0xD0, 0xD8)):
            continue
        if position + 2 > len(data):
            break
        length = int.from_bytes(data[position : position + 2], "big")
        end = position + length
        if length < 2 or end > len(data):
            break
        result.append((start, end, marker, position + 2))
        position = end
        if marker == 0xDA:
            # Entropy-coded data escapes FF as FF00; restart markers have no payload.
            while True:
                position = data.find(b"\xff", position)
                if position < 0:
                    raise ValueError("JPEG image has no end marker")
                following = position + 1
                while following < len(data) and data[following] == 255:
                    following += 1
                if following >= len(data):
                    raise ValueError("Truncated JPEG scan")
                code = data[following]
                if code == 0 or 0xD0 <= code <= 0xD7:
                    position = following + 1
                else:
                    break
    raise ValueError("Truncated JPEG image")


def xmp_segments(data):
    spans, _ = segments(data)
    return [
        data[start:end]
        for start, end, marker, payload in spans
        if marker == 0xE1 and data[payload:end].startswith(XMP_HEADERS)
    ]


def mpf_entries(packet, payload_offset):
    """Locate MPF image records within an APP2 packet's TIFF directory."""
    base = payload_offset + 4  # skip MPF signature
    data = packet[base:]
    if data[:4] not in (b"II\x2a\0", b"MM\0\x2a"):
        raise ValueError("Unsupported MPF directory")
    order = "<" if data[:2] == b"II" else ">"
    try:
        directory = struct.unpack_from(order + "I", data, 4)[0]
        count = struct.unpack_from(order + "H", data, directory)[0]
        for index in range(count):
            tag, kind, length, offset = struct.unpack_from(
                order + "HHII", data, directory + 2 + index * 12
            )
            if tag == 0xB002:
                if kind != 7 or length < 16 or length % 16 or offset + length > len(data):
                    raise ValueError("Invalid MPF image list")
                return order, base, [base + offset + n for n in range(0, length, 16)]
    except struct.error as error:
        raise ValueError("Truncated MPF directory") from error
    raise ValueError("MPF image list is missing")


def preserve_jpeg_packets(source: Path, destination: Path):
    original, output = source.read_bytes(), destination.read_bytes()
    source_spans, source_end = segments(original)
    output_spans, output_end = segments(output)
    additions = []
    for start, end, marker, payload in source_spans:
        if marker == 0xE1 and original[payload:end].startswith(XMP_HEADERS):
            additions.append((original[start:end], None))
        elif marker == 0xE2 and original[payload:end].startswith(b"MPF\0"):
            packet = original[start:end]
            additions.append((packet, (payload - start, start)))
    removed = []
    for start, end, marker, payload in output_spans:
        if (marker == 0xE1 and output[payload:end].startswith(XMP_HEADERS)) or (
            marker == 0xE2 and output[payload:end].startswith(b"MPF\0")
        ):
            removed.append((start, end))
    chunks = []
    previous = 2
    for start, end in removed:
        chunks.append(output[previous:start])
        previous = end
    chunks.append(output[previous:output_end])
    body = b"".join(chunks)
    # Keep EXIF ahead of MPF: ExifTool's binary EXIF extraction uses the first
    # TIFF directory encountered. Insert packets before the first image scan.
    body_spans, _ = segments(b"\xff\xd8" + body)
    insertion = next(start for start, _, marker, _ in body_spans if marker == 0xDA) - 2
    primary_length = 2 + sum(len(packet) for packet, _ in additions) + len(body)
    fixed = []
    packet_start = 2 + insertion
    for packet, mpf in additions:
        if mpf:
            payload, old_start = mpf
            order, base, entries = mpf_entries(packet, payload)
            mutable = bytearray(packet)
            for index, entry in enumerate(entries):
                size, offset = struct.unpack_from(order + "II", packet, entry + 4)
                if index == 0:
                    if offset != 0:
                        raise ValueError("Unsupported MPF primary image offset")
                    struct.pack_into(order + "II", mutable, entry + 4, primary_length, 0)
                else:
                    old_absolute = old_start + base + offset
                    if old_absolute < source_end or old_absolute + size > len(original):
                        raise ValueError("MPF auxiliary image points outside the JPEG trailer")
                    new_absolute = primary_length + old_absolute - source_end
                    struct.pack_into(
                        order + "I", mutable, entry + 8, new_absolute - (packet_start + base)
                    )
            packet = bytes(mutable)
        fixed.append(packet)
        packet_start += len(packet)
    rebuilt = (
        b"\xff\xd8" + body[:insertion] + b"".join(fixed) + body[insertion:] + original[source_end:]
    )
    destination.write_bytes(rebuilt)
    if xmp_segments(original) != xmp_segments(rebuilt):
        raise ValueError("JPEG XMP packet verification failed; no output was published")
    if rebuilt[primary_length:] != original[source_end:]:
        raise ValueError("JPEG auxiliary data verification failed; no output was published")
