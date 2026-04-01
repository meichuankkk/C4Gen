import os
import shutil

# 定义源目录和目标目录
source_dir = '/root/autodl-tmp/view1/enre_py_reports/java_reports'
target_dir = os.path.join(source_dir, 'hudi')

# 如果目标目录不存在，则创建
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# 获取源目录下的所有文件
files = os.listdir(source_dir)

moved_count = 0
for file in files:
    file_path = os.path.join(source_dir, file)
    
    # 检查是否为文件，并且文件名包含 'hudi' (排除文件夹自身)
    if os.path.isfile(file_path) and 'hudi' in file and file.endswith('.json'):
        target_path = os.path.join(target_dir, file)
        shutil.move(file_path, target_path)
        moved_count += 1
        print(f"Moved: {file}")

print(f"-"*30)
print(f"完成。共移动了 {moved_count} 个 hudi 文件到 {target_dir}")
