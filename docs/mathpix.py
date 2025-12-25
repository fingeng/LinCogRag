from mpxpy.mathpix_client import MathpixClient
import os
import time
from pathlib import Path
# app_id="physvistaproject_7743c4_7b0256",
# app_key="78c9209455088a0717a039436047a0a2d0bd49bfd1c0eed285f10a935fe7e095"
os.environ['MATHPIX_APP_ID']="physvistaproject_7743c4_7b0256"
os.environ['MATHPIX_APP_KEY']="589b94eaa5dea9da0ad538c09f78168e25390c753d79d157593aa731547aca86"
def pdf_to_html(pdf_path, output_dir="output"):
    """
    将本地PDF转换为HTML格式（兼容版）
    Args:
        pdf_path: 本地PDF文件路径
        output_dir: 输出目录路径
    Returns:
        html_path: 生成的HTML文件路径
    """
    # 初始化客户端
    client = MathpixClient(
        app_id=os.getenv("MATHPIX_APP_ID"),
        app_key=os.getenv("MATHPIX_APP_KEY")
    )
    
    # 检查输入文件
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF文件不存在: {pdf_path}")
    
    os.makedirs(output_dir, exist_ok=True)

    try:
        # 新版API调用方式（移除了options参数）
        pdf = client.pdf_new(
            file_path=pdf_path,
            convert_to_html=True  # 只保留必要参数
        )

        # 等待处理完成
        print("正在处理PDF...")
        start_time = time.time()
        timeout = 300  # 5分钟超时
        
        while True:
            status = pdf.pdf_status()
            if status.get('status') == 'completed':
                break
            elif time.time() - start_time > timeout:
                raise TimeoutError("处理超时")
            time.sleep(5)
            print(f"当前状态: {status.get('status')}")

        # 保存结果
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        html_path = os.path.join(output_dir, f"{base_name}.html")
        pdf.to_html_file(path=html_path)
        
        print(f"转换成功！HTML文件已保存到: {html_path}")
        return html_path

    except Exception as e:
        print(f"处理失败: {str(e)}")
        raise


def pdf_to_md(input_path, output_dir="output", output_name=None, timeout=300):
    """
    将PDF/图片转换为Markdown（基础版）
    
    Args:
        input_path: 输入文件路径（PDF或图片）
        output_dir: 输出目录
        output_name: 输出文件名（不含扩展名），如果为None则使用输入文件名
        timeout: 处理超时时间（秒）
    
    Returns:
        Tuple(md_path, md_content)
    """
    # 初始化客户端（建议从环境变量读取密钥）
    client = MathpixClient(
        app_id=os.getenv("MATHPIX_APP_ID"),
        app_key=os.getenv("MATHPIX_APP_KEY")
    )
    
    # 确保输出目录存在
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        # 提交转换任务
        print(f"正在处理: {Path(input_path).name}")
        result = client.pdf_new(
            file_path=input_path,
            convert_to_md=True  # 关键参数：启用Markdown转换
        )
        
        # 等待处理完成
        if not result.wait_until_complete(timeout=timeout):
            raise TimeoutError(f"处理超时（{timeout}秒）")
        
        # 生成输出路径
        if output_name is None:
            output_name = Path(input_path).stem
        md_path = Path(output_dir) / f"{output_name}.md"
        
        # 获取Markdown内容并保存
        md_content = result.to_md_text()
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        print(f"✓ 转换成功: {md_path}")
        return md_path, md_content
        
    except Exception as e:
        print(f"✗ 转换失败 {Path(input_path).name}: {str(e)}")
        raise RuntimeError(f"转换失败: {str(e)}")


def batch_pdf_to_md(directory, output_dir=None, name_mapping=None, timeout=300):
    """
    批量将目录下的PDF文件转换为Markdown
    
    Args:
        directory: 包含PDF文件的目录路径
        output_dir: 输出目录，如果为None则使用输入目录
        name_mapping: PDF文件名到英文简写的映射字典，如果为None则自动生成
        timeout: 每个文件的处理超时时间（秒）
    """
    directory = Path(directory)
    if output_dir is None:
        output_dir = directory
    else:
        output_dir = Path(output_dir)
    
    # 默认文件名映射（中文名 -> 英文简写）
    default_mapping = {
        "物理竞赛真题解析 热学 光学 近代物理.pdf": "phy_comp_thermal_optics",
        "物理竞赛解题方法漫谈.pdf": "phy_comp_methods",
        "荣誉物理 力学部分.pdf": "honor_phy_mechanics",
        "荣誉物理 热学、光学、近代物理部分.pdf": "honor_phy_thermal_optics",
        "荣誉物理-电学部分.pdf": "honor_phy_electromagnetism",
        "高中物理奥林匹克竞赛标准教材 郑永令.pdf": "hs_phy_olympiad_standard",
        "高中物理奥赛方法(清晰版).pdf": "hs_phy_olympiad_methods",
    }
    
    if name_mapping is None:
        name_mapping = default_mapping
    
    # 查找所有PDF文件
    pdf_files = list(directory.glob("*.pdf"))
    
    if not pdf_files:
        print(f"在目录 {directory} 中未找到PDF文件")
        return
    
    print(f"找到 {len(pdf_files)} 个PDF文件，开始批量转换...")
    print("-" * 60)
    
    success_count = 0
    failed_files = []
    
    for i, pdf_file in enumerate(pdf_files, 1):
        pdf_name = pdf_file.name
        print(f"\n[{i}/{len(pdf_files)}] 处理文件: {pdf_name}")
        
        # 获取对应的英文简写名称
        output_name = name_mapping.get(pdf_name)
        if output_name is None:
            # 如果没有映射，使用原文件名（去除扩展名）
            output_name = pdf_file.stem
            print(f"  警告: 未找到映射，使用原文件名: {output_name}")
        
        try:
            pdf_to_md(
                input_path=str(pdf_file),
                output_dir=str(output_dir),
                output_name=output_name,
                timeout=timeout
            )
            success_count += 1
        except Exception as e:
            print(f"  错误: {str(e)}")
            failed_files.append(pdf_name)
        
        # 在文件之间稍作延迟，避免API限流
        if i < len(pdf_files):
            time.sleep(2)
    
    print("\n" + "=" * 60)
    print(f"批量转换完成！")
    print(f"成功: {success_count}/{len(pdf_files)}")
    if failed_files:
        print(f"失败的文件:")
        for f in failed_files:
            print(f"  - {f}")
    print("=" * 60)

if __name__ == "__main__":
    try:
        # 替换为您的本地PDF路径
        input_pdf = r"/home/maoxy23/projects/LinearRAG/docs/2511.13201v1.pdf"  
        output_folder = "/home/maoxy23/projects/LinearRAG/docs/benchmarks"
        
        # # result_html = pdf_to_html(input_pdf, output_folder)
        # # print(f"最终输出文件: {result_html}")
        # output_folder = r"/mnt/c/Users/X_fig/Desktop/Ubuntu/Dataset/competition_new/exp9.md"
        
        result_md = pdf_to_md(input_pdf, output_folder)
        # print(f"最终输出文件: {result_md}")
        #batch_pdf_to_md(r"/home/maoxy23/projects/LinearRAG/docs", output_dir=r"/home/maoxy23/projects/LinearRAG/docs/benchmarks", timeout=300)
        
    except Exception as e:
        print(f"执行出错: {e}")