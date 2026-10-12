"""Write plan_ra29.sha256 (sha256 and file name of plan_ra29.md). Run once, after the plan is final."""
import hashlib
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
p = os.path.join(HERE, "plan_ra29.md")
h = hashlib.sha256(open(p, "rb").read()).hexdigest()
open(os.path.join(HERE, "plan_ra29.sha256"), "w", newline="\n").write("%s  plan_ra29.md\n" % h)
print(h)
