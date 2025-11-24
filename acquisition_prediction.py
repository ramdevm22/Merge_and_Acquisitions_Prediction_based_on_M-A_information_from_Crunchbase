
import argparse
from scripts.feature_extraction import do_feature_extraction
from scripts.model import do_model_building

p = argparse.ArgumentParser()
p.add_argument("--extract", action="store_true")
p.add_argument("--fit", action="store_true")
args = p.parse_args()

if args.extract:
    do_feature_extraction()
if args.fit:
    do_model_building()
if not (args.extract or args.fit):
    p.print_help()
