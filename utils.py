# -*- coding: utf-8 -*-

import psutil

def estimate_max_jobs(bytes_per_worker: float) -> int:
    avail_bytes = psutil.virtual_memory().available
    safe_bytes = avail_bytes * 0.8
    return max(1, int(safe_bytes // bytes_per_worker))

def format_time(seconds: float) -> str:
    h, m = divmod(seconds, 3600)
    m, s = divmod(m, 60)
    if h:
        return f"{int(h)}h {int(m)}m {s:.2f}s"
    elif m:
        return f"{int(m)}m {s:.2f}s"
    else:
        return f"{s:.2f}s"

