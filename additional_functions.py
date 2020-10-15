import csv

def create_csv(file, fieldnames):
    with open(file, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
def write_csv(file, info):
    fieldnames = info.keys()
    with open(file, 'a') as f:
        csv_writer = csv.DictWriter(f, fieldnames=fieldnames)
        csv_writer.writerow(info)
