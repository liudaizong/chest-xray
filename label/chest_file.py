import csv
import pickle

def nestedlist2csv(list, out_file):
    with open(out_file, 'wb') as f:
        w = csv.writer(f)
        fieldnames=list[0].keys()  # solve the problem to automatically write the header
        w.writerow(fieldnames)
        for row in list:
            w.writerow(row.values())

def csv_reorder(in_file, out_file,lst_order):
    with open(in_file, 'rb') as infile, open(out_file, 'wb') as outfile:
        fieldnames=lst_order
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in csv.DictReader(infile):
            writer.writerow(row)

with open('xray.csv','rb') as csvfile:
	reader = csv.DictReader(csvfile)
	dicts = [row for row in reader]

with open('test_list.txt','r') as f:
	data_list = f.readlines()
	data_list = [i.split('\n')[0] for i in data_list]

label = []
for num in range(len(dicts)):
	label.append({})

step = 0
how = 0
disease_list = ['Atelectasis','Cardiomegaly','Effusion','Infiltration','Mass','Nodule','Pneumonia','Pneumothorax']
for column in range(len(dicts)):
	if dicts[column]['Image Index'] == data_list[how]:
		label[step]['Image Index'] = dicts[column]['Image Index']
		for disease in disease_list:
			if disease in dicts[column]['Finding Labels']:
				label[step][disease] = 1
			else:
				label[step][disease] = 0
		# if label[step]['Image Index'] == '00001335_006.png':
		# 	break
		step += 1
		how += 1
	if how == len(data_list):
		break

nestedlist2csv(label,'test.csv')
print(len(label))
lst_order = ['Image Index','Atelectasis','Cardiomegaly','Effusion','Infiltration','Mass','Nodule','Pneumonia','Pneumothorax']
csv_reorder('test.csv','test_l.csv',lst_order)