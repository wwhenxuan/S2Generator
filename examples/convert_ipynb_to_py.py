import os
import argparse
import nbformat
from nbconvert import PythonExporter


def convert_ipynb_to_py(ipynb_path, py_path):
    """将单个IPython Notebook文件转换为Python脚本"""
    try:
        # 读取notebook文件
        with open(ipynb_path, "r", encoding="utf-8") as f:
            notebook = nbformat.read(f, as_version=4)

        # 转换为Python代码
        exporter = PythonExporter()
        source, _ = exporter.from_notebook_node(notebook)

        # 保存为Python文件
        with open(py_path, "w", encoding="utf-8") as f:
            f.write(source)

        print(f"转换成功: {ipynb_path} -> {py_path}")
        return True

    except Exception as e:
        print(f"转换失败 {ipynb_path}: {str(e)}")
        return False


def convert_all_ipynb_in_folder(folder_path, output_folder=None):
    """转换文件夹中的所有IPython Notebook文件"""
    if not os.path.exists(folder_path):
        print(f"错误: 文件夹路径不存在 '{folder_path}'")
        return

    # 如果没有指定输出文件夹，则使用同一文件夹
    if output_folder is None:
        output_folder = folder_path
    else:
        os.makedirs(output_folder, exist_ok=True)

    # 获取所有.ipynb文件
    ipynb_files = [f for f in os.listdir(folder_path) if f.endswith(".ipynb")]

    if not ipynb_files:
        print("在指定文件夹中未找到任何.ipynb文件")
        return

    print(f"找到 {len(ipynb_files)} 个IPython Notebook文件")

    # 转换每个文件
    success_count = 0
    for filename in ipynb_files:
        ipynb_path = os.path.join(folder_path, filename)
        py_filename = filename.replace(".ipynb", ".py")
        py_path = os.path.join(output_folder, py_filename)

        if convert_ipynb_to_py(ipynb_path, py_path):
            success_count += 1

    print(f"转换完成: 成功 {success_count}/{len(ipynb_files)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="将文件夹中的所有IPython Notebook文件转换为Python脚本"
    )
    parser.add_argument("folder", nargs="?", default=".", help="要处理的文件夹路径（默认为当前目录）")
    parser.add_argument("-o", "--output", help="输出文件夹路径（默认为同一文件夹）")

    args = parser.parse_args()

    convert_all_ipynb_in_folder(args.folder, args.output)


# # 转换并将结果保存到指定输出目录
# python convert_ipynb_to_py.py /path/to/notebooks -o /path/to/output
