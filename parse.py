from pathlib import Path
import fitz
import re
import ast
import pandas as pd
from pipeline_iter8 import analysis

all_pdfs = list(Path("iter8_pdfs").glob("*.pdf"))

for pdf_path in all_pdfs[3:4]:
    variables = {}

    with fitz.open(pdf_path) as doc:
        text = "\n".join(page.get_text() for page in doc)

    for name in ["global_data", "regional_data", "product_data"]:
        match = re.search(
            rf'{name}\s*=\s*(\{{.*?\}}|\[.*?\])',
            text,
            flags=re.DOTALL,
        )
        if match is None:
            match = re.search(
            rf'"{name}"\s*:\s*(\{{.*?\}}|\[.*?\])',
            text,
            flags=re.DOTALL,
        )
        if match:
            variables[name] = ast.literal_eval(match.group(1))


    global_data = variables.get("global_data")
    regional_data = variables.get("regional_data")
    product_data = variables.get("product_data")
    product_name = product_data[0]["product"]


    global_df = pd.DataFrame([global_data])
    regional_df = pd.DataFrame(regional_data)
    product_df = pd.DataFrame(product_data)

    print(product_name)
    print(analysis(global_df, regional_df, product_df, product_name))
