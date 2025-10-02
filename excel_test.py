import pandas as pd
from openpyxl import Workbook

wb = Workbook()
ws = wb.active
ws['A1'] = "sas"
ws.append([5,4,3])
dic = {
    "col1", 9,
    "col2", 10,
    "col3", 11,
}
# ws.append(dic)
wb.save("sample.xlsx")