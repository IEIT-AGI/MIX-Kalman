def remove_None_value_elements(input_dict):
    if type(input_dict) is not dict:
        return None
    result = {}
    for key in input_dict:
        tmp = {}
        if input_dict[key] is not None:
            if type(input_dict[key]).__name__ == 'dict':
                tmp.update({key: remove_None_value_elements(input_dict[key])})
            else:
                tmp.update({key: input_dict[key]})
        result.update(tmp)
    return result
