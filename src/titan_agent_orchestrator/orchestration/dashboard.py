"""Kế thừa bộ lọc từ kết quả dashboard đã lưu trong hội thoại."""

import re
import unicodedata


def normalized(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text.casefold().replace("đ", "d")) if unicodedata.category(c) != "Mn")


def dashboard_arguments(message: str, previous: dict | None = None) -> dict | None:
    content = normalized(message)
    content = re.sub(r"\b(?:builng|builing)\b", "building", content)
    content = re.sub(r"\bcong to(?: do)?\b", "thiet bi do", content)
    if any(term in content for term in ("tieu thu", "so sanh", "xu huong", "thang nay", "thang truoc")):
        return None
    changing = any(term in content for term in ("doi ", "chuyen ", "loc ", "chi xem", "tat ca", "bo loc"))
    explicit_building_edit = changing and "toa nha" in content
    if not ("dashboard" in content or (previous and changing) or explicit_building_edit):
        return None
    if not previous and not any(term in content for term in ("bms", "building", "toa nha", "thiet bi", "diem do", "overview", "tong quan")):
        return None
    arguments = dict((previous or {}).get("filters", {}))
    if any(term in content for term in ("xoa bo loc", "tat ca toa nha", "all buildings")):
        return {}
    options = (previous or {}).get("options", {})
    labels = (previous or {}).get("option_labels", {}).get("building", {})
    named_buildings = {
        identity: label.split(" · ")[0]
        for identity, label in labels.items()
        if re.search(r"(?<!\w)" + re.escape(normalized(label.split(" · ")[0])) + r"(?!\w)", content)
    }
    named_building = None
    if named_buildings:
        longest = max(len(normalized(name)) for name in named_buildings.values())
        named_buildings = {identity: name for identity, name in named_buildings.items() if len(normalized(name)) == longest}
        previous_source = str(arguments.get("building", "")).partition(":")[0]
        same_source = [identity for identity in named_buildings if identity.partition(":")[0] == previous_source]
        if len(same_source) == 1:
            named_building = same_source[0]
        elif len(named_buildings) == 1:
            named_building = next(iter(named_buildings))
        else:
            # Tên giống nhau ở nhiều nguồn: lọc theo tên, không chọn tùy tiện một ID.
            named_building = next(iter(named_buildings.values()))
        arguments.pop("equipment", None)
        arguments["building"] = named_building
    for key in ("building", "equipment", "equipment_type", "object_type", "status", "unit"):
        matches = [value for value in options.get(key, []) if re.search(r"(?<!\w)" + re.escape(normalized(value)) + r"(?!\w)", content)]
        if matches:
            value = max(matches, key=len)
            if key == "building" and value != arguments.get(key):
                arguments.pop("equipment", None)
            arguments[key] = value
    building_edit = "toa nha" in content or "thiet bi" not in content
    if building_edit and not named_building and ("building" not in arguments or not any(normalized(v) in content for v in options.get("building", []))):
        full_name = re.search(r"(?:tòa nhà|toa nha)\s*[:：]?\s*(?:(?:sang|là|la)\s+)?(.+?)[.!?]*$", message, re.IGNORECASE)
        if full_name and full_name.group(1).strip():
            arguments.pop("equipment", None)
            term = full_name.group(1).strip()
            arguments["building"] = "Building " + term.upper() if re.fullmatch(r"[a-zA-Z]", term) else term
            return arguments
        match = re.search(r"(?:toa nha\s*[:：]?\s+(?:building\s+)?|building\s+)([a-z0-9_-]+)", content)
        if match and match.group(1) not in {"nao", "do", "theo"}:
            arguments.pop("equipment", None)
            arguments["building"] = "Building " + match.group(1).upper()
    if "doi thiet bi" in content and not any(normalized(v) in content for v in options.get("equipment", [])):
        match = re.search(r"thiet bi(?: do)?\s+(?:sang\s+)?(.+?)[.!?]*$", content)
        if match:
            arguments["equipment"] = match.group(1).strip()
    return arguments


def dashboard_route(message: str, previous_capability: str | None, previous: dict | None):
    """Chọn dashboard theo yêu cầu mới hoặc kế thừa dashboard gần nhất."""
    text = normalized(message)
    overview = "overview" in text or "tong quan" in text
    comparison = "so sanh" in text and any(term in text for term in ("dien", "nuoc", "electricity", "water"))
    changing = any(term in text for term in ("doi ", "chuyen ", "loc ", "chi xem", "tat ca", "bo loc"))
    utility_change = any(term in text for term in ("nuoc", "water", "dien", "electricity")) and changing
    if comparison or (previous_capability == "fm.get_energy_comparison" and (changing or utility_change) and not overview):
        context = previous if previous_capability == "fm.get_energy_comparison" else None
        filters = dashboard_arguments(message, context) if changing and not comparison else dict((context or {}).get("filters", {}))
        if filters is None:
            filters = dict((context or {}).get("filters", {}))
        metric = "water" if any(term in text for term in ("nuoc", "water")) else "electricity" if any(term in text for term in ("dien", "electricity")) else None
        if metric and metric != filters.get("metric", "electricity"):
            filters.pop("equipment", None)
            filters.pop("object_type", None)
        if metric:
            filters["metric"] = metric
        for term, grouping in (("theo toa nha", "building"), ("theo thiet bi", "equipment"), ("theo cong to", "equipment"), ("theo object type", "object_type")):
            if term in text:
                filters['group_by'] = grouping
        if comparison:
            match = re.search(r"(?:toa nha\s+|building\s+)(?:building\s+)?([ab])\b", text)
            if match:
                filters['building'] = 'Building ' + match.group(1).upper()
                filters.pop('equipment', None)
        for kind in ("Energy", "Consumption"):
            if re.search(r"object type\s*[:：]?\s*" + kind.casefold() + r"\b", text):
                filters['object_type'] = kind
        meter = str(filters.get("equipment", ""))
        if re.fullmatch(r"[a-zA-Z]", meter):
            matches = [choice for choice in (context or {}).get("options", {}).get("equipment", []) if choice.casefold().endswith("-" + meter.casefold())]
            if len(matches) == 1:
                filters["equipment"] = matches[0]
        return "fm.get_energy_comparison", {k: v for k, v in filters.items() if k in {"building", "equipment", "metric", "object_type", "group_by"}}
    if overview or "dashboard" in text or (previous_capability == "fm.get_bms_dashboard" and changing):
        context = previous if previous_capability == "fm.get_bms_dashboard" and not overview else None
        arguments = dashboard_arguments(message, context)
        if arguments is not None:
            return "fm.get_bms_dashboard", arguments
    return None
