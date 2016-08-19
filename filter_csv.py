import csv

from progress.bar import Bar

def filter(e):
    return e.replace('\n', ' ')

path = '/home/djstrong/projects/repos/cycloped-io/cycloped-io/statistical_classification/multi_5first_paragraphs_features_class.csv'

with open(path+'.filtered','w') as f2:
    writer = csv.writer(f2)
    with open(path) as f:
        reader = csv.reader(f)
        bar = Bar('Processing', suffix='%(index)d/%(max)d %(percent).1f%% - %(eta_td)s', max=2799601)
        for row in reader:
            bar.next()
            row2 = [filter(e) for e in row]
            writer.writerow(row2)
        bar.finish()