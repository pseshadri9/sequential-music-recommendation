
'''
Load data from file and return pandas/np array
containing unaltered data
'''
def load_data(filepath):
    pass

'''
Process loaded dataset into desirable format/etc.
'''
def process_data(data):
    pass

'''
Partition data into desired samples, etc. 
for train/test/val
'''
def partition_data(data):
    pass


'''
wrapper function to return usable dataset from filepath
'''

def create_dataset(filepath):
    data = load_data(filepath)
    data = process_data(data)

    return data