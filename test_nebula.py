#!/usr/bin/env python3
"""Direct text/image diagnostics using the perception server's transport; no ROS."""

import argparse
from functools import partial
import json
import os
from pathlib import Path
import time

from ros2.cmoc_perception.cmoc_perception.vlm_http import (
    NEBULA_URL, NEBULA_MODEL, ask_nebula, post_json, safe_detail, encode_nebula_image,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default=NEBULA_URL)
    parser.add_argument('--model', default=NEBULA_MODEL)
    parser.add_argument('--image', type=Path, default=Path(__file__).parent / 'test.png')
    parser.add_argument('--image-max-dimension', type=int, default=1024)
    parser.add_argument('--jpeg-quality', type=int, default=85)
    args = parser.parse_args()
    key = os.environ.get('NEBULA_API_KEY')
    print('Endpoint:', safe_detail(args.url, key), flush=True)
    print('Model:', safe_detail(args.model, key), flush=True)
    print('NEBULA_API_KEY present:', bool(key), flush=True)
    if not key:
        print('NEBULA_API_KEY is not set; export it before running this script.')
        return 1
    import cv2
    pixels = cv2.imread(str(args.image))
    if pixels is None:
        raise ValueError(f'Could not read diagnostic image: {args.image}')
    encoded = encode_nebula_image(
        pixels, max_dimension=args.image_max_dimension, jpeg_quality=args.jpeg_quality,
        log_info=print)
    root = Path(__file__).parent
    schema = json.loads((root / 'schemas/perception/create_scene_graph.json').read_text())
    schema['properties']['objects']['items'] = json.loads((root / 'schemas/objects.json').read_text())
    failures = 0
    for kind, picture, requested_schema, prompt in (
        ('text', None, None, 'Reply with the word OK.'),
        ('image', encoded, None, 'Briefly describe what is visible in this image.'),
        ('image + observation schema', encoded, schema, 'Describe visible entities and relations. Omit unknown qualities.'),
    ):
        print('Request type:', kind, flush=True)
        started = time.monotonic()
        try:
            response = ask_nebula(prompt, picture, requested_schema, api_key=key,
                                 url=args.url, model=args.model,
                                 post=partial(post_json, timeout=120), log_info=print)
            print('Success:', isinstance(response, str), flush=True)
        except Exception as error:
            failures += 1
            print('Error:', safe_detail(getattr(error, 'detail', str(error)), key), flush=True)
        print(f'Elapsed: {time.monotonic() - started:.2f}s', flush=True)
    return int(bool(failures))


if __name__ == '__main__':
    raise SystemExit(main())
