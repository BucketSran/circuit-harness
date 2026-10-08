"""Minimal polygon/label fixture converter. Replace with the lab's generator."""

import json
import math
import re
import sys

import gdstk


def convert(value, output):
    if set(value) - {"cell", "polygons", "labels"} or not re.fullmatch(
        r"[A-Za-z_][A-Za-z0-9_]{0,31}", value["cell"]
    ):
        raise ValueError("unsupported layout schema or cell name")
    library = gdstk.Library(unit=1e-6, precision=1e-9)
    cell = library.new_cell(value["cell"])
    if not value["polygons"]:
        raise ValueError("at least one polygon is required")
    for polygon in value["polygons"]:
        if set(polygon) != {"points", "layer", "datatype"}:
            raise ValueError("polygon requires points, layer and datatype")
        points = polygon["points"]
        if not 3 <= len(points) <= 10000 or any(
            len(p) != 2
            or any(type(n) not in (int, float) or not math.isfinite(n) or abs(n) > 1e6 for n in p)
            for p in points
        ):
            raise ValueError("invalid polygon coordinates in micrometers")
        for key in ("layer", "datatype"):
            if type(polygon[key]) is not int or not 0 <= polygon[key] <= 65535:
                raise ValueError("layer/datatype must be integers in [0,65535]")
        cell.add(gdstk.Polygon(points, layer=polygon["layer"], datatype=polygon["datatype"]))
    for label in value.get("labels", []):
        if set(label) != {"text", "origin", "layer", "texttype"}:
            raise ValueError("invalid label schema")
        if not isinstance(label["text"], str) or not label["text"] or len(label["text"]) > 128:
            raise ValueError("invalid label text")
        if len(label["origin"]) != 2 or any(
            type(n) not in (int, float) or not math.isfinite(n) or abs(n) > 1e6
            for n in label["origin"]
        ):
            raise ValueError("invalid label origin")
        if any(
            type(label[k]) is not int or not 0 <= label[k] <= 65535 for k in ("layer", "texttype")
        ):
            raise ValueError("invalid label layer/texttype")
        cell.add(gdstk.Label(**label))
    library.write_gds(output)


if __name__ == "__main__":
    with open(sys.argv[1]) as stream:
        convert(json.load(stream), sys.argv[2])
