from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent


class PathConfig:
    DATA_DIR = ROOT_DIR / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    DEFAULT_DATA_FILE = DATA_DIR / "Medicaldataset.csv"
    MODEL_DIR = ROOT_DIR / "models"
    ASSETS_DIR = ROOT_DIR / "assets"

    @classmethod
    def ensure_dirs(cls) -> None:
        for path in (
            cls.DATA_DIR,
            cls.RAW_DATA_DIR,
            cls.PROCESSED_DATA_DIR,
            cls.MODEL_DIR,
            cls.ASSETS_DIR,
        ):
            path.mkdir(parents=True, exist_ok=True)


class AppConfig:
    PROJECT_NAME = "基于机器学习的心脏病数据分析与预测系统"
    PAGE_TITLE = PROJECT_NAME
    PAGE_ICON = "❤️"
    LAYOUT = "wide"
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    TARGET_COLUMN = "Result"
    POSITIVE_LABEL = "positive"
    NEGATIVE_LABEL = "negative"


FEATURE_LABELS = {
    "Age": "年龄",
    "Gender": "性别",
    "Heart rate": "心率",
    "Systolic blood pressure": "收缩压",
    "Diastolic blood pressure": "舒张压",
    "Blood sugar": "血糖",
    "CK-MB": "CK-MB",
    "Troponin": "肌钙蛋白",
    "Cholesterol": "胆固醇",
    "Max HR": "最大心率",
    "ST depression": "ST段压低",
    "Chest pain type": "胸痛类型",
    "Exercise angina": "运动心绞痛",
    "Number of vessels": "血管狭窄数",
    "Thallium scan": "铊扫描",
    "Result": "诊断结果",
}
