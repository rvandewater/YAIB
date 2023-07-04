class DataSplit:
    train = "train"
    val = "val"
    test = "test"


class DataSegment:
    static = "STATIC"
    dynamic = "DYNAMIC"
    outcome = "OUTCOME"  # Labels
    features = "FEATURES"  # Combined features from static and dynamic data.


class VarType:
    group = "GROUP"
    sequence = "SEQUENCE"
    label = "LABEL"

class FeatType:
    FEAT_NAMES = ['s_cat' , 's_cont' , 'k_cat' , 'k_cont' , 'o_cat' , 'o_cont' , 'target', 'id']
