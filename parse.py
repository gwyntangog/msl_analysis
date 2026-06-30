from pathlib import Path
import fitz
import re
import ast
import pandas as pd
from pipeline_iter8 import analysis

all_pdfs = list(Path("iter8_pdfs").glob("*.pdf"))
# print(all_pdfs)
for pdf_path in all_pdfs[:1]:
    variables = {}
    # print(pdf_path)
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
    # print(type(global_data))
    # print(type(regional_data))
    # print(type(product_data))
    # print(product_data[0]["product"])
    product_name = product_data[0]["product"]


    global_df = pd.DataFrame([global_data])
    regional_df = pd.DataFrame(regional_data)
    product_df = pd.DataFrame(product_data)
    # print(global_df)

# from pipeline_iter8 import analysis

    print(analysis(global_df, regional_df, product_df, product_name))

# from pathlib import Path
# import fitz

# def extract_block(text, key):
#     start = text.find(key)
#     if start == -1:
#         return None

#     start = text.find("{", start)
#     if start == -1:
#         return None

#     depth = 0
#     for i in range(start, len(text)):
#         if text[i] == "{":
#             depth += 1
#         elif text[i] == "}":
#             depth -= 1
#             if depth == 0:
#                 return text[start:i+1]
#     return None


# def normalize(block):
#     if block is None:
#         return None
#     return block.replace(",\n}", "\n}").replace(",}", "}")


# def parse(block):
#     import ast
#     try:
#         return ast.literal_eval(block)
#     except:
#         return None


# variables = {}

# for pdf_path in Path("iter8_pdfs").glob("*.pdf"):
#     with fitz.open(pdf_path) as doc:
#         text = "\n".join(page.get_text() for page in doc)

#     pdf_vars = {}

#     for name in ["global_data", "regional_data", "product_data"]:
#         raw = extract_block(text, name)
#         cleaned = normalize(raw)
#         parsed = parse(cleaned)

#         if parsed is not None:
#             pdf_vars[name] = parsed

#     variables[pdf_path.stem] = pdf_vars
#     global_data = variables[pdf_path.stem].get("global_data")
#     regional_data = variables[pdf_path.stem].get("regional_data")
#     product_data = variables[pdf_path.stem].get("product_data")
#     # print(global_data)
#     # print(type(global_data))
#     global_df = pd.DataFrame([global_data])
#     global_df.to_csv("test_global_parse")
#     # print(global_df)
#     regional_df = pd.DataFrame([regional_data])
#     regional_df.to_csv("test_regional_parse")
#     # print(regional_df)
#     product_df = pd.DataFrame([product_data])
#     product_df.to_csv("test_product_parse")
#     # print(product_df)
#     product_name = product_df["product"].iloc[0]

#     print(product_name)


#     analysis(global_df, regional_df, product_df, product_name)
