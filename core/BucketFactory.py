# implementation based on https://github.com/irhete/predictive-monitoring-benchmark and https://github.com/nirdizati/nirdizati-training-backend

import EncoderFactory

from bucketers.PrefixLengthBucketer import PrefixLengthBucketer
from bucketers.ZeroBucketer import ZeroBucketer

        
def get_bucketer(method, case_id_col=None):

    if method == "zero":
        return ZeroBucketer(case_id_col=case_id_col)

    elif method == "prefix":
        return PrefixLengthBucketer(case_id_col=case_id_col)

    else:
        print("Invalid bucketer type")
        return None