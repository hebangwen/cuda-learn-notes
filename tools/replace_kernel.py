import os
import shutil

import argparse
import re


def replace_cuda_kernel(file_content):
    # 使用正则表达式定位内核函数
    pattern = re.compile(
        r"(__global__\s+void\s+\w+\(.*?\)\s*)\{.*?\}(?=\n\s*\n)",
        re.DOTALL
    )
    replacement = r"\1{\n    // [START MANUAL IMPLEMENTATION]\n    // TODO: 请在此实现内核代码\n    // [END MANUAL IMPLEMENTATION]\n}"

    # 替换内核实现部分
    processed_code = re.sub(pattern, replacement, file_content)

    return processed_code


def parse_args():
    parser = argparse.ArgumentParser(description="convert implmented cuda kernel file into no implementation file by regex")
    parser.add_argument("input_file", type=str, help="input cuda file")
    parser.add_argument("--inplace", action="store_true", help="modify file in place")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # 处理文件名
    if args.inplace:
        base_name = os.path.splitext(args.input_file)[0]
        ref_file = f"{base_name}.ref.cu"
        output_file = args.input_file

        # 创建备份文件
        with open(args.input_file, "r") as f:
            content = f.read()
            with open(ref_file, "w") as rf:
                rf.write(content)
    else:
        output_file = args.output_file

    with open(args.input_file, "r") as f:
        processed_code = replace_cuda_kernel(f.read())

    with open(output_file, "w") as of:
        of.write(processed_code)

    print(f"write replaced file into {output_file}")
