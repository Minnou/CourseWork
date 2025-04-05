import re
import os

def split_by_year(filename="./dataset.csv", result_folder=""):
    """Split dataset by years creating multiple csv files

    Args:
        filename: path to the original dataset
        result_folder: path to the folder where result files will be stored
    Returns:
        True if successful False if not
    """
    file = open(filename, "r")
    dates = []
    values = []
    current_year = ""
    try:
        for line in file:
            if __name__ == '__main__':
                print("Текущая строка: " + line)
            date_and_value = line.strip().split(';')
            year = re.match(pattern="(.*)-.*-.*", string=date_and_value[0]).group(1)
            if year == None:
                if __name__ == '__main__':
                    print("Выбран неподходящий файл. Конец.")
                return False
            if current_year == "":
                current_year = year
            if current_year != year:
                current_year = year
                if result_folder != "":
                    try:
                        if not (os.path.isdir(result_folder)):
                            os.mkdir(result_folder)
                    except Exception as e:
                        if __name__ == '__main__':
                            print("Что-то пошло не так:\n" + e.__class__.__name__)
                        return False
                else:
                    result_folder = "./"
                if not result_folder.endswith("/"):
                    result_folder = result_folder + "/"
                result_file = open(file=result_folder + str(dates[len(dates) - 1]).replace("-", "") + "_" + str(dates[0]).replace("-", "") +".csv", mode="w+")  
                for i in range(0, len(dates)):
                    result_file.write(str(dates[i]) + ";" + str(values[i]) +"\n")
                result_file.close()
                dates.clear()
                values.clear()
            dates.append(date_and_value[0])
            values.append(date_and_value[1])
    except KeyboardInterrupt:
        pass
    file.close()
    return True

def main():
    print("Введите путь до файла (./dataset.csv): ", end="")
    filename = input()
    if filename.strip() == "":
        filename = "./dataset.csv"
    
    print("Введите путь до папки, куда будут сохранены результаты (./by_year): ", end="")
    result_folder = input()
    if result_folder.strip() == "":
        result_folder = "./by_year"

    if split_by_year(filename=filename, result_folder=result_folder):
        print("Успех")
    else:
        print("Провал")
    
if __name__ == '__main__':
    main()