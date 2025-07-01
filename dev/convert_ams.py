import os

import pandas as pd
import yulon_template as YL


class AMSConverter:
    """Class for convert other brand AMS to YL AMS."""

    def __init__(self):
        self.script_dir = os.path.dirname(__file__)
        self.info = pd.read_csv(os.path.join(self.script_dir, "data", "info.csv"))

    def convert_ams(self, uid, input_path, output_path, intermediate_folder="tmp"):
        """Convert other brand AMS to YL AMS.

        Args:
            uid (str): info.csv中的uid，對應到檔案該如何處理(main workspace位置、車型、OCR範圍等資訊)
            input_path (str): 他廠AMS檔案，需為 PDF 格式 (以.pdf結尾)
            output_path (str): 裕隆AMS，以.xlsx結尾，若有多頁則會於結尾添加 (1), (2), ...等字串
            intermediate_folder (str): 中途產生的檔案所放置的資料夾
        """
        if intermediate_folder:
            intermediate_folder = os.path.abspath(intermediate_folder)
            os.makedirs(intermediate_folder, exist_ok=True)

        ams_info = self.info[self.info["uid"] == uid].iloc[0]

        main_workspace_paths = YL.data_utils.extract_main_workspace_from_pdf(
            input_path,
            intermediate_folder,
            YL.data_utils.to_int_list(ams_info["main_workspace_pos"]),
            [1409, 836],
        )  # size without right bar: (1091, 836)

        tables = YL.data_utils.extract_table_from_pdf(input_path, ams_info)

        """班站工順, 工序編號, 版次 因目前未知需後續處理"""
        for i, table in enumerate(tables):
            process = YL.YulonProcess(table["process"]["id"], table["process"]["name"])
            revise_records = [
                YL.YulonReviseRecord(record["date"], record["record"])
                for record in table["revise_records"]
            ]
            template = YL.YulonTemplate(
                ams_info["type"],
                "",
                process,
                revise_records,
                int(table["page"]),
                int(table["last_page"]),
                table["refer_graph"],
                "",
            )

            if len(main_workspace_paths) > i:
                template.image_dict = {"B5": main_workspace_paths[i]}

            template.export_xlsx(
                output_path[:-5] + ("" if i == 0 else str(i)) + ".xlsx"
            )


"""Testing code"""
converter = AMSConverter()
converter.convert_ams(
    "T33-1",
    "data/T33/N/pdf/FOP_10400-XNP_N_SET-DRAIN HOSE TO BODY e-POWER_BROUILLON_(e-POWER).pdf",
    "dev/data/result.xlsx",
    "dev/tmp",
)
# converter.convert_ams("D31-1", "data/D/D/pdf/C10070 D31 BEV 裝底鈑下皮塞-3.pdf", "dev/data/result.xlsx", "dev/tmp")
