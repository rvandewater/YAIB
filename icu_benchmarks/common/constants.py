# Label specific constants
MORTALITY_NAME = "Mortality_At24Hours"
CIRC_FAILURE_NAME = "Dynamic_CircFailure"
RESP_FAILURE_NAME = "Dynamic_RespFailure"
URINE_REG_NAME = "Dynamic_UrineOutput_2Hours_Reg"
URINE_BINARY_NAME = "Dynamic_UrineOutput_2Hours_Binary"
PHENOTYPING_NAME = "Phenotyping_APACHEGroup"
LOS_NAME = "Remaining_LOS_Reg"

FILE_NAMES = {
    "DYNAMIC": "dyn.parquet",
    "OUTCOME": "outc.parquet",
    "STATIC": "sta.parquet",
}
VARS = {
    "STAY_ID": "stay_id",
    "TIME": "time",
    "SEX": "sex",
    "DYNAMIC_VARS": ["dbp", "hr", "map", "o2sat", "resp", "sbp", "temp"],
    "STATIC_VARS": ["age", "sex", "height", "weight"],
}
