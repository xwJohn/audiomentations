import os
import shutil

output_dir = 'output'

#src_list = ['ch','pf','wm','zy','xw','pg']
src_list = ['old']
rec_files = ["开灯","关灯","绿色","蓝色","红色","温度","湿度","气压","亮度"]


for name in rec_files:
    in_folder = output_dir + '/' + name
    os.makedirs(in_folder,exist_ok=True)

for name in src_list:
    output_src = 'output_' + name
    #print(output_src)
    for root,dirs,files in os.walk(output_src):
        new_foler = os.path.join(root).replace(output_src,'output') 
        print(new_foler)       
        for filename in files:
            new_name = filename.replace(name,name + '_')
            whole_name = os.path.join(root,filename)
            whole_newname = os.path.join(root,new_name)
            os.rename(whole_name, whole_newname)
            shutil.move(whole_newname,new_foler)




