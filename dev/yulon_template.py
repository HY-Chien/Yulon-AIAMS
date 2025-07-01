import os
import shutil

import data_utils


class YulonProcess:
    """Class for yulon process.

    Args:
        id (str): 工序編號
        process_name (dict[str]): 工序名稱，"ch"為中文，"en"為英文
    """

    def __init__(self, id="", process_name: dict[str] = None):
        self.id = id
        self.name = {"ch": "", "en": ""} if process_name is None else process_name


class YulonReviseRecord:
    """Class for yulon revise record.

    Args:
        date (str): 修訂紀錄的日期(年/月/日)
        record (str): 修訂紀錄的記事
    """

    def __init__(self, date="", record=""):
        self.date = date
        self.record = record


class YulonTemplate:
    """Class for yulon template.

    Args:
        type (int): YL車型，YLM-XXX的"XXX"部分
        route (str): 班站/工順
        process (YulonProcess): 工序，包含編號、中文名稱、英文名稱，參考 `YulonProcess`
        revise_records (list[YulonReviseRecord]): 修訂紀錄，參考 `YulonReviseRecord`
        page (int): 頁次
        last_page (int): 最後一頁的頁次
        refer_graph (str): 參考圖面
        edition (str): 版次
        image_dict (dict[str,str]): 圖片資訊，格式為 {cell: path_to_image}，例如 {"B2": "path/to/image.png"}
    """

    def __init__(
        self,
        type,
        route="",
        process: YulonProcess = None,
        page=1,
        last_page=1,
        refer_graph="",
        edition="",
        main_workspace_path=None,
        revise_records: list[YulonReviseRecord] = None,
        image_dict: dict[str, str] = None,
    ):
        self.script_dir = os.path.dirname(__file__)
        self.type = type
        self.route = route
        self.process = YulonProcess() if process is None else process
        self.revise_records = revise_records or []
        self.page = page
        self.last_page = last_page
        self.refer_graph = refer_graph
        self.edition = edition
        self.main_workspace_path = main_workspace_path
        self.image_dict = image_dict or {}

    def get_text_dict(self):
        self.text_dict = {
            "D2": self.type,
            "B44": self.route,
            "D44": self.process.id,
            "G42": self.process.name["ch"],
            "G45": self.process.name["en"],
            "L48": self.refer_graph,
            "T48": self.edition,
            "Z48": str(self.page),
            "AB48": str(self.last_page),
        }

        date_pos = ["J44", "J46", "T42", "T44", "T46"]
        record_pos = ["M44", "M46", "W42", "W44", "W46"]
        for i, record in enumerate(self.revise_records):
            self.text_dict[date_pos[i]] = record.date
            self.text_dict[record_pos[i]] = record.record

        return self.text_dict

    def get_image_dict(self):
        if self.main_workspace_path is not None:
            self.image_dict["B5"] = self.main_workspace_path

        return self.image_dict

    def export_xlsx(self, output_path: str):
        """Export current template to xlsx file.

        Args:
            output_path (str): 檔案儲存位置，須以 ".xlsx" 結尾
        """
        # if file_utils.get_file_extension(output_path) is not 'xlsx':
        #   print("File extension error. Should be xlsx")
        #   return

        shutil.copyfile(
            os.path.join(self.script_dir, "data/template.xlsx"), output_path
        )
        data_utils.write_xlsx(output_path, self.get_text_dict(), self.image_dict)
