import json


def data_dump(filename, data, mode='a'):
    json_str = json.dumps(data, indent=4)
    with open(filename, mode) as json_file:
        json_file.write(json_str)
        json_file.close()
