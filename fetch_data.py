#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import io
import sys
import tarfile
import zipfile
import gzip
import shutil
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse, unquote
from urllib.request import Request, urlopen

# ===================== 配置 =====================
TARGET_ROOT = os.path.abspath(".")  # 根目录（每个数据集会在这里创建一个子文件夹）
TIMEOUT = 60
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)

# 15 个数据集（文件夹名 -> UCI 数据集页面 URL）
DATASETS = {
    # 1) Molecular Biology (Splice-junction Gene Sequences)
    "molecular_biology_splice": "https://archive.ics.uci.edu/dataset/69/molecular+biology+splice+junction+gene+sequences",
    # 2) Iris
    "iris": "https://archive.ics.uci.edu/ml/datasets/iris",
    # 3) Hepatitis
    "hepatitis": "https://archive.ics.uci.edu/dataset/46/hepatitis",
    # 4) Planning Relax
    "planning_relax": "https://archive.ics.uci.edu/dataset/230/planning+relax",
    # 5) Parkinsons
    "parkinsons": "https://archive.ics.uci.edu/ml/datasets/parkinsons",
    # 6) Sonar (Connectionist Bench – Mines vs. Rocks)
    "sonar": "https://archive.ics.uci.edu/ml/datasets/connectionist%2Bbench%2B%28sonar%2C%2Bmines%2Bvs.%2Brocks%29",
    # 7) Glass Identification
    "glass_identification": "https://archive.ics.uci.edu/dataset/42/glass%2Bidentification",
    # 8) Audiology (Original)
    "audiology_original": "https://archive.ics.uci.edu/dataset/7/audiology%2Boriginal",
    # 9) SPECTF Heart
    "spectf_heart": "https://archive.ics.uci.edu/ml/datasets/SPECTF%2BHeart",
    # 10) Breast Cancer Wisconsin (Original)
    "breast_cancer_wisconsin_original": "https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original",
    # 11) Haberman’s Survival
    "haberman_survival": "https://archive.ics.uci.edu/ml/datasets/Haberman",
    # 12) Dermatology
    "dermatology": "https://archive.ics.uci.edu/dataset/33/dermatology",
    # 13) Indian Liver Patient (ILPD)
    "indian_liver_patient_ilpd": "https://archive.ics.uci.edu/ml/datasets/ILPD%2B%28Indian%2BLiver%2BPatient%2BDataset%29",
    # 14) Tic-Tac-Toe Endgame
    "tic_tac_toe_endgame": "https://archive.ics.uci.edu/dataset/101/tic%2Btac%2Btoe%2Bendgame",
    # 15) Image Segmentation
    "image_segmentation": "https://archive.ics.uci.edu/ml/datasets/image%2Bsegmentation",
}

# 尝试下载的文件后缀
DOWNLOAD_EXTS = {
    ".zip", ".tar", ".tgz", ".tar.gz", ".gz",
    ".csv", ".data", ".txt", ".names", ".dat", ".arff",
    ".xls", ".xlsx"
}

# 自动解压的压缩包后缀
ARCHIVE_EXTS = {".zip", ".tar", ".tgz", ".tar.gz", ".gz"}

# ==================================================

def http_get(url: str) -> bytes:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=TIMEOUT) as r:
        return r.read()

class LinkExtractor(HTMLParser):
    def __init__(self, base_url: str):
        super().__init__()
        self.base = base_url
        self.links = []  # (absolute_href)

    def handle_starttag(self, tag, attrs):
        if tag.lower() != "a":
            return
        href = None
        for k, v in attrs:
            if k.lower() == "href":
                href = v
                break
        if href:
            abs_href = urljoin(self.base, href)
            self.links.append(abs_href)

def find_links(html: bytes, base_url: str):
    parser = LinkExtractor(base_url)
    try:
        parser.feed(html.decode("utf-8", errors="ignore"))
    except Exception:
        pass
    return parser.links

def is_probable_data_folder(href: str) -> bool:
    # 新老站两种常见数据目录路径
    h = href.lower()
    return ("/ml/machine-learning-databases/" in h) or ("/static/public/" in h)

def looks_like_download(href: str) -> bool:
    path = urlparse(href).path.lower()
    if any(path.endswith(ext) for ext in DOWNLOAD_EXTS):
        return True
    # 某些直链含查询参数
    if any(href.lower().split("?")[0].endswith(ext) for ext in DOWNLOAD_EXTS):
        return True
    return False

def dedupe(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out

def sanitize_filename(name: str) -> str:
    # 去掉查询与片段；清理特殊字符
    name = unquote(name.split("?")[0].split("#")[0])
    name = re.sub(r"[^\w.\-()+=, ]", "_", name)
    return name

def ensure_dir(path: str):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)

def choose_unique_path(directory: str, base_name: str) -> str:
    """如果已存在同名文件，则在文件名后增加 (1), (2) ...，避免覆盖。"""
    # 处理 .tar.gz / .tar.bz2 等双扩展
    root = base_name
    ext = ""
    if base_name.lower().endswith(".tar.gz"):
        root = base_name[:-7]
        ext = ".tar.gz"
    elif base_name.lower().endswith(".tar.bz2"):
        root = base_name[:-8]
        ext = ".tar.bz2"
    else:
        root, ext = os.path.splitext(base_name)

    p = os.path.join(directory, root + ext)
    k = 1
    while os.path.exists(p):
        p = os.path.join(directory, f"{root}({k}){ext}")
        k += 1
    return p

def save_bytes(path: str, content: bytes):
    with open(path, "wb") as f:
        f.write(content)

def extract_archive(path: str, target_dir: str) -> bool:
    lower = path.lower()
    try:
        if lower.endswith(".zip"):
            with zipfile.ZipFile(path, "r") as zf:
                zf.extractall(target_dir)
            return True
        elif lower.endswith(".tar") or lower.endswith(".tar.gz") or lower.endswith(".tgz"):
            mode = "r:gz" if (lower.endswith(".tar.gz") or lower.endswith(".tgz")) else "r"
            with tarfile.open(path, mode) as tf:
                tf.extractall(target_dir)
            return True
        elif lower.endswith(".gz") and not lower.endswith(".tar.gz") and not lower.endswith(".tgz"):
            # 单文件 .gz：解压为同名无 .gz 的文件
            out_path = re.sub(r"\.gz$", "", path, flags=re.IGNORECASE)
            with gzip.open(path, "rb") as gzf, open(out_path, "wb") as out:
                shutil.copyfileobj(gzf, out)
            return True
        else:
            return False
    except Exception as e:
        print(f"[WARN] 解压失败 {path}: {e}")
        return False

def crawl_dataset_page(dataset_url: str):
    print(f"  - 抓取页面: {dataset_url}")
    try:
        html = http_get(dataset_url)
    except Exception as e:
        print(f"    [ERROR] 打开页面失败: {e}")
        return []

    links = find_links(html, dataset_url)
    hrefs = [u for u in links]

    # 优先“数据文件夹”
    data_folders = [u for u in hrefs if is_probable_data_folder(u)]
    direct_files = [u for u in hrefs if looks_like_download(u)]

    # 进入数据文件夹里再找一层
    for dfurl in data_folders:
        print(f"    · 数据文件夹: {dfurl}")
        try:
            df_html = http_get(dfurl)
            df_links = find_links(df_html, dfurl)
            for href in df_links:
                if looks_like_download(href):
                    direct_files.append(href)
        except Exception as e:
            print(f"      [WARN] 打不开数据文件夹: {e}")

    direct_files = dedupe(direct_files)
    if not direct_files:
        print("    [INFO] 未发现可直接下载的文件链接（可能需要手动）。")
    else:
        for f in direct_files:
            print(f"    + 发现文件: {f}")
    return direct_files

def download_and_extract(url: str, target_dir: str):
    ensure_dir(target_dir)
    # 选择文件名
    path = urlparse(url).path
    base = sanitize_filename(os.path.basename(path) or "download")
    out_path = choose_unique_path(target_dir, base)

    try:
        print(f"      ↓ 下载: {os.path.basename(out_path)}")
        data = http_get(url)
        save_bytes(out_path, data)
    except Exception as e:
        print(f"      [ERROR] 下载失败: {e}")
        return

    # 如果是压缩包 -> 解压并删除
    lowered = out_path.lower()
    is_archive = any(lowered.endswith(ext) for ext in ARCHIVE_EXTS)
    if is_archive:
        ok = extract_archive(out_path, target_dir)
        if ok:
            try:
                os.remove(out_path)
                print(f"      ✓ 解压完成并删除压缩包: {os.path.basename(out_path)}")
            except Exception as e:
                print(f"      [WARN] 解压后无法删除压缩包: {e}")
        else:
            print(f"      [WARN] 未能解压: {os.path.basename(out_path)}")
    else:
        print(f"      ✓ 保存完成: {os.path.basename(out_path)}")

def main():
    print(f"目标根目录: {TARGET_ROOT}")
    for folder, page in DATASETS.items():
        dataset_dir = os.path.join(TARGET_ROOT, folder)
        ensure_dir(dataset_dir)
        print(f"\n=== 数据集 [{folder}] ===")
        files = crawl_dataset_page(page)
        if not files:
            continue
        for f in files:
            download_and_extract(f, dataset_dir)

    print("\n[DONE] 所有数据集处理完成（压缩包已就地解压，原包已删除）。")

if __name__ == "__main__":
    main()