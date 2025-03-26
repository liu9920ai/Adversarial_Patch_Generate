import json
import shutil
import os
import re

# ��ȡ����mod��Ϣ���ļ�
factorio_path = 'E:\SteamLibrary\steamapps\common\Factorio'#�˴���Ϊfactorio�İ�װ·��'
mod_path = factorio_path + '\\'+ 'mods'#�˴���Ϊfactorio mod�Ĵ���ļ���

# Ŀ���ļ���·��
destination_folder = factorio_path + '\\active_mod_save'#�˴���Ϊ������factorio mod�Ĵ��·��

json_path = mod_path + '\\' + 'mod-list.json'
if os.path.exists(destination_folder) == False:
    os.makedirs(destination_folder)
with open(json_path, 'r') as file:
    mods_data = json.load(file)

# ģʽ������ȡ�汾��
pattern = re.compile(r'(.+)_\d+\.\d+\.\d+\.zip')
        
# ����mods
for mod in mods_data['mods']:
    if mod['enabled'] == True:
        mod_name = mod['name']
        
        # ����mod�ļ�����ͬһ���ļ�����
        source_path = f'{mod_path}/{mod_name}_*.zip'
        mod_files = [f for f in os.listdir(mod_path) if re.match(f"{mod_name}_\d+\.\d+\.\d+\.zip", f)]
           
        if mod_files:
            for mod_file in mod_files:
                shutil.copy(os.path.join(mod_path, mod_file), os.path.join(destination_folder, mod_file))
                print(f'mod {mod_file} copied')
#��json�ļ�Ҳ���ƹ�ȥ
shutil.copy(json_path, destination_folder + '\\' + 'mod-list.json')