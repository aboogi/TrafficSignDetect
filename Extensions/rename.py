import os

list_files = os.listdir('DS_TS_label')
txt_files = list()
for file in list_files:
    exchange = file.split('.')[1]  #os.path.split(file)[1]
    if exchange == 'txt':
        txt_files.append(file)

for txt_file in txt_files:
    with open(os.path.join('DS_TS_label', txt_file), 'r+') as label:
        lines_file = list()
        for line in label:
            if line[0] == '6':
                new_line = '4' + line[1:]
                nline = line.replace(line, new_line)
                lines_file.append(nline)
                with open(os.path.join('DS_TS_label', txt_file), 'w+') as label_write:
                    for lin in lines_file:
                        label_write.write(lin)
            else: 
                lines_file.append(line)
    
