import json
import os
import re


_HASH_SUFFIX_RE = re.compile(r"^(?P<name>.+)_[0-9a-f]{6,}$")
_LEADING_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}-")
_VERSION_PREFIX_RE = re.compile(r"-v(?=\d)")


def _normalize_package_name(value: str) -> str:
    """Normalize a path or file name to the feature file base name."""
    base_name = os.path.basename(value).lower()
    if base_name.endswith(".tar.gz"):
        base_name = base_name[:-7]
    elif base_name.endswith(".tgz"):
        base_name = base_name[:-4]
    elif base_name.endswith(".tar"):
        base_name = base_name[:-4]
    elif base_name.endswith(".zip"):
        base_name = base_name[:-4]
    elif base_name.endswith(".csv"):
        base_name = base_name[:-4]
    match = _HASH_SUFFIX_RE.match(base_name)
    if match:
        base_name = match.group("name")
    base_name = _LEADING_DATE_RE.sub("", base_name)
    base_name = _VERSION_PREFIX_RE.sub("-", base_name)
    return base_name


def load_malicious_type_map(groundtruth_path: str):
    """Load malicious type mappings from a jsonl groundtruth file.

    Returns:
        (type_map, primary_type_map) keyed by feature file base name.
    """
    type_map = {}
    primary_type_map = {}
    with open(groundtruth_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            record = json.loads(line)
            key = None
            for field in ("dst_path", "src_archive", "archive_name", "package_name", "name"):
                field_value = record.get(field)
                if field_value:
                    key = _normalize_package_name(field_value)
                    break
            if not key:
                record_id = record.get("id")
                if record_id is not None:
                    key = str(record_id)
            if not key:
                continue

            annotation = record.get("annotation") or {}
            verdict = annotation.get("verdict")
            if verdict and str(verdict).lower() != "malicious":
                continue

            malicious_types = annotation.get("malicious_types")
            if malicious_types is None:
                malicious_types = record.get("malicious_types") or []
            if isinstance(malicious_types, str):
                malicious_types = [malicious_types]
            type_bucket = annotation.get("type_bucket") or record.get("type_bucket")

            if type_bucket and not malicious_types:
                malicious_types = [type_bucket]
            if malicious_types:
                malicious_types = list(dict.fromkeys(malicious_types))

            primary_type = type_bucket or (malicious_types[0] if malicious_types else "UNKNOWN")
            type_map[key] = malicious_types
            primary_type_map[key] = primary_type
    return type_map, primary_type_map
