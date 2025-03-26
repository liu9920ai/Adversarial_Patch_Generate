import json
import shutil
import os
import re

# 读取包含mod信息的文件
factorio_path = 'E:\SteamLibrary\steamapps\common\Factorio'#此处改为factorio的安装路径'
mod_path = factorio_path + '\\'+ 'mods'#此处改为factorio mod的存放文件夹

# 目标文件夹路径
destination_folder = factorio_path + '\\active_mod_save'#此处改为挑出的factorio mod的存放路径

json_path = mod_path + '\\' + 'mod-list.json'
if os.path.exists(destination_folder) == False:
    os.makedirs(destination_folder)
with open(json_path, 'r') as file:
    mods_data = json.load(file)

# 模式用于提取版本号
pattern = re.compile(r'(.+)_\d+\.\d+\.\d+\.zip')
        
# 遍历mods
for mod in mods_data['mods']:
    if mod['enabled'] == True:
        mod_name = mod['name']
        
        # 假设mod文件都在同一个文件夹中
        source_path = f'{mod_path}/{mod_name}_*.zip'
        mod_files = [f for f in os.listdir(mod_path) if re.match(f"{mod_name}_\d+\.\d+\.\d+\.zip", f)]
           
        if mod_files:
            for mod_file in mod_files:
                shutil.copy(os.path.join(mod_path, mod_file), os.path.join(destination_folder, mod_file))
                print(f'mod {mod_file} copied')
#将json文件也复制过去
shutil.copy(json_path, destination_folder + '\\' + 'mod-list.json')