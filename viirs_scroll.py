"""
Script to generate a VIIRS RGB of the closest granule to a provided time
"""
from pathlib import Path
from datetime import datetime as dt
from datetime import timedelta as td
from pprint import pprint as ppt

from aes670hw2 import viirs
from aes670hw2 import laads

debug = True
#target_time = dt.utcnow()
target_time = dt(year=2020, month=3, day=14)
time_radius = td(days=1)
buffer_dir = Path("data/buffer")
satellite = "NP"
bands = (16, 15, 12, 10, 9, 7, 5, 4, 3)
image_dimensions = (1080, 1920)
token="eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJBUFMgT0F1dGgyIEF1dGhlbnRpY2F0b3IiLCJpYXQiOjE2Nzg0MDEyNzgsIm5iZiI6MTY3ODQwMTI3OCwiZXhwIjoxNjkzOTUzMjc4LCJ1aWQiOiJtZG9kc29uIiwiZW1haWxfYWRkcmVzcyI6Im10ZDAwMTJAdWFoLmVkdSIsInRva2VuQ3JlYXRvciI6Im1kb2Rzb24ifQ.gwlWtdrGZ1CNqeGuNvj841SjnC1TkUkjxb6r-w4SOmk"

granules = viirs.query_viirs_l1b(
        product_key=f"V{satellite}02MOD",
        start_time=target_time-time_radius,
        end_time=target_time+time_radius,
        add_geo=False,
        debug=debug
        )

granules.sort(key=lambda g: (abs(g["atime"]-target_time)).total_seconds())

gran_dict = granules[0]
gran_path = laads.download(
        target_url=gran_dict["downloadsLink"],
        dest_dir=buffer_dir,
        raw_token=token,
        replace=True,
        debug=debug
        )

is_day = gran_dict["illuminations"] == "D"
half_int = lambda r: int(r/2)
data, info = viirs.get_viirs_data(
        l1b_file=gran_path, bands=bands, debug=debug)

center = list(map(half_int, data.shape))
yrange, xrange = map(
        lambda c, dc: (c-half_int(dc), c+half_int(dc)),
        zip(center, image_dimensions)
        )

