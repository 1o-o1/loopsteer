import json, sys
d = json.load(open(sys.argv[1]))
ks = ("gate_zero_edit", "gate_placement", "gate_cache_equiv", "gate_identities")
ok = all((d.get(k) or {}).get("pass") for k in ks)
print({k: (d.get(k) or {}).get("pass") for k in ks})
sys.exit(0 if ok else 3)
