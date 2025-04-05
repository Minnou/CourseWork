def make_x_y(filename="./dataset.csv", x_file ="./x.csv", y_file = "./y.csv"):
    """Create x and y csv files from the original dataset

    Args:
        filename: path to the original dataset
        x_file: path to x file
        y_file: path to y file
    Returns:
        True if successful False if not
    """
    file = open(filename, "r")
    dates_file = open(x_file, "w+")
    values_file = open(y_file, "w+")
    try:
        for line in file:
            if __name__ == '__main__':
                print("Текущая строка: " + line)
            date_and_value = line.strip().split(';')
            if len(date_and_value) != 2:
                if __name__ == '__main__':
                    print("Выбран неподходящий файл. Конец.")
                return False
            dates_file.write(date_and_value[0]+"\n")
            values_file.write(date_and_value[1]+"\n")
    except KeyboardInterrupt:
        pass
    file.close()
    dates_file.close()
    values_file.close()
    return True

def main():
    print("Введите путь до файла (./dataset.csv): ", end="")
    filename = input()
    if filename.strip() == "":
        filename = "./dataset.csv"
    
    print("Введите путь до файла с датами (./x.csv): ", end="")
    x_file = input()
    if x_file.strip() == "":
        x_file = "./x.csv"
    
    print("Введите путь до файла с ценами (./y.csv): ", end="")
    y_file = input()
    if y_file.strip() == "":
        y_file = "./y.csv"
    
    if make_x_y(filename=filename, x_file=x_file, y_file=y_file):
         print("Успех")
    else:
        print("Провал")

if __name__ == '__main__':
    main()