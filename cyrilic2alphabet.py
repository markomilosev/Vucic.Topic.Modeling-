
def transcribe(a):

#string = "ће сутра поднети потписе сто посланика владајуће већине за одржавање саветодавног референдума о поверењу председнику"
    cyrilic = ['а', 'б', 'в', 'г', 'д', 'ђ', 'е', 'ж', 'з', 'и', 'ј', 'к', 'л', 'љ', 'м', 'н', 'њ', 'о', 'п', 'р', 'с', 'т', 'ћ', 'у', 'ф', 'х', 'ц', 'ч', 'џ', 'ш']
    alphabet = ['a', 'b', 'v', 'g', 'd', 'đ', 'e', 'ž', 'z', 'i', 'j', 'k', 'l', 'lj', 'm', 'n', 'nj', 'o', 'p', 'r', 's', 't', 'ć', 'u', 'f', 'h', 'c', 'č', 'dž', 'š']

    replace_map = {cyrilic[i]: alphabet[i] for i in range(len(cyrilic))}

    result = ''.join(
        replace_map.get(char.lower(), char).upper() if char.isupper() else replace_map.get(char.lower(), char)
        for char in a)

    return result