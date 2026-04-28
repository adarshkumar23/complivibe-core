import json
from pathlib import Path
from collections import Counter

def normalize_who(val: str) -> str:
    if not val:
        return "unknown"
    v = val.lower()
    if "provider" in v or "manufacturer" in v:
        return "provider"
    elif "deploy" in v or "employer" in v or "user" in v:
        return "deployer"
    elif "importer" in v:
        return "importer"
    elif "distributor" in v:
        return "distributor"
    elif "fiduciary" in v or "controller" in v:
        return "data_fiduciary"
    elif "processor" in v:
        return "data_processor"
    elif any(k in v for k in ["authority", "commission", "notified", "member state", "board", "office", "enisa", "member states"]):
        return "competent_authority"
    elif "all" in v or "any" in v or "everyone" in v:
        return "all"
    return val

def reclassify_type(what: str, current_type: str) -> str:
    if current_type != "other":
        return current_type
    if not what:
        return current_type
        
    w = what.lower()
    if any(k in w for k in ["document", "record", "log"]):
        return "documentation"
    elif any(k in w for k in ["notify", "report", "inform"]):
        return "notification"
    elif any(k in w for k in ["register", "list", "database"]):
        return "registration"
    elif any(k in w for k in ["transparent", "explain", "disclose"]):
        return "transparency"
    elif any(k in w for k in ["risk", "assess", "evaluat"]):
        return "risk_management"
    elif any(k in w for k in ["human", "oversight", "review"]):
        return "human_oversight"
    elif any(k in w for k in ["prohibit", "banned", "forbidden"]):
        return "prohibited"
    elif any(k in w for k in ["data", "training", "dataset"]):
        return "data_governance"
    elif any(k in w for k in ["accurate", "robust", "test"]):
        return "accuracy_robustness"
    elif "consent" in w:
        return "consent"
    elif any(k in w for k in ["right", "access", "erase"]):
        return "rights"
    elif any(k in w for k in ["secur", "protect", "encrypt"]):
        return "security"
    
    return "other"

def process_file(file_path: Path):
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return
        
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    before_who_stats = Counter()
    after_who_stats = Counter()
    before_type_stats = Counter()
    after_type_stats = Counter()
    
    high_conf_data = []
    low_conf_data = []
    
    for item in data:
        who = item.get("who_must_comply", "")
        item["who_must_comply_original"] = who
        before_who_stats[who] += 1
        
        normalized_who = normalize_who(who)
        item["who_must_comply"] = normalized_who
        after_who_stats[normalized_who] += 1
        
        obl_type = item.get("obligation_type", "")
        before_type_stats[obl_type] += 1
        
        what = item.get("what_must_be_done", "")
        reclassified_type = reclassify_type(what, obl_type)
        item["obligation_type"] = reclassified_type
        after_type_stats[reclassified_type] += 1
        
        # Confidence filtering
        confidence = item.get("confidence")
        if confidence is not None and float(confidence) < 0.5:
            low_conf_data.append(item)
        else:
            high_conf_data.append(item)
        
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(high_conf_data, f, indent=2, ensure_ascii=True)
        
    if low_conf_data:
        low_conf_path = file_path.parent / f"{file_path.stem}_low_confidence.json"
        with open(low_conf_path, "w", encoding="utf-8") as f:
            json.dump(low_conf_data, f, indent=2, ensure_ascii=True)
        
    print(f"--- Stats for {file_path.name} ---")
    print(f"Total Obligations Initially: {len(data)}")
    print(f"Removed Low Confidence (<0.5): {len(low_conf_data)}")
    print(f"Kept High Confidence (>=0.5): {len(high_conf_data)}")
    print("\nwho_must_comply (Top 5 Before -> Top 5 After):")
    for (bk, bv), (ak, av) in zip(before_who_stats.most_common(5), after_who_stats.most_common(5)):
        print(f"  {bk}: {bv}   ->   {ak}: {av}")
        
    print("\nobligation_type (Top 5 Before -> Top 5 After):")
    for (bk, bv), (ak, av) in zip(before_type_stats.most_common(5), after_type_stats.most_common(5)):
        print(f"  {bk}: {bv}   ->   {ak}: {av}")
    print("\n")


def main():
    root_dir = Path(__file__).parent.parent
    files_to_process = [
        root_dir / "data" / "processed" / "eu_ai_act_obligations.json",
        root_dir / "data" / "processed" / "dpdp_obligations.json"
    ]
    
    for file_path in files_to_process:
        process_file(file_path)

if __name__ == "__main__":
    main()
