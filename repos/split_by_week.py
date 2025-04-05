import os
import datetime

def split_by_week(filename="./dataset.csv", result_folder=""):
    """Split dataset by weeks creating multiple csv files

    Args:
        filename: path to the original dataset
        result_folder: path to the folder where result files will be stored
    Returns:
        True if successful False if not
    """
    file = open(filename, "r")
    dates = []
    values = []
    last_day_of_week = -1
    try:
        for line in file:
            if __name__ == '__main__':
                print("Текущая строка: " + line)
            date_and_value = line.strip().split(';')
            if len(date_and_value) != 2:
                if __name__ == '__main__':
                    print("Выбран неподходящий файл. Конец.")
                return False
            date = date_and_value[0]
            date = datetime.datetime.strptime(date, "%Y-%m-%d")
            current_day_of_week = date.weekday() #Понедельник 0 Воскресенье 6
            if last_day_of_week < current_day_of_week and last_day_of_week != -1:
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
            last_day_of_week = current_day_of_week
    except KeyboardInterrupt:
        pass
    file.close()
    return True

def main():
    print("Введите путь до файла (./dataset.csv): ", end="")
    filename = input()
    if filename.strip() == "":
        filename = "./dataset.csv"
    
    print("Введите путь до папки, куда будут сохранены результаты (./by_week): ", end="")
    result_folder = input()
    if result_folder.strip() == "":
        result_folder = "./by_week"

    if split_by_week(filename=filename, result_folder=result_folder):
        print("Успех")
    else:
        print("Провал")
    
if __name__ == '__main__':
    main()